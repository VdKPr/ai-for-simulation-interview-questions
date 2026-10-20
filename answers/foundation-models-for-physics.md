# Foundation Models for Physics

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A physics foundation model is pretrained on a large, diverse corpus of simulations (many PDEs, geometries, regimes or both) so that it can be used **zero-shot** or **fine-tuned with little data** for new tasks. Research models for PDEs (**MPP**, **DPOT** with up to 0.5 billion parameters, **Poseidon**) show that pretraining reduces the data needed downstream, sometimes by orders of magnitude, and that the gains grow with model and data size (Subramanian et al.). Companies now offer pretrained engineering models: PhysicsX's **LGM-Aero** ("Large Geometry Model", trained on more than 25 million meshes and tens of thousands of CFD and FEA simulations), Luminary's **SHIFT-SUV** and **SHIFT-Wing**, and NVIDIA's DoMINO-based services. **Weather** is the existence proof: GraphCast, GenCast and ECMWF's operational AIFS learned from decades of reanalysis and now rival or beat physics-based forecasts. What transfers well: geometry encoders, generic flow features and good initializations; what transfers badly: new physics, new boundary-condition types, regimes outside the corpus and new output fields. **Fine-tuning on a small customer dataset** needs a grouped validation split, strong baselines and honest reporting, because fine-tuning can also hurt (negative transfer). Whether **data is the moat** is debated: high-fidelity data is expensive and durable, but open datasets grow, customer data belongs to customers, and workflow integration and trust may matter as much.

**On this page**

- [What a PDE foundation model is and what transfers](#what-a-pde-foundation-model-is-and-what-transfers)
- [Large geometry and large engineering models](#large-geometry-and-large-engineering-models)
- [Lessons from ML weather forecasting](#lessons-from-ml-weather-forecasting)
- [Fine-tuning on a small customer dataset](#fine-tuning-on-a-small-customer-dataset)
- [Is data the moat?](#is-data-the-moat)
- [Code: when pretraining helps, and when it hurts](#code-when-pretraining-helps-and-when-it-hurts)
- [In an interview](#in-an-interview)

## What a PDE foundation model is and what transfers

**The recipe.** Collect trajectories or solutions from many PDE families (fluid dynamics, reaction-diffusion, waves, magnetohydrodynamics), with different parameters, resolutions and boundary conditions; train one large model (usually a transformer operating on patches or tokens) to predict the next state or the solution; then fine-tune on a downstream task with few examples, or use it directly.

**Research models.**

| Model | Key ideas | Reported findings |
| --- | --- | --- |
| **Multiple Physics Pretraining (MPP)**, McCabe et al. (Polymathic AI), 2023 | Autoregressive pretraining of a transformer on several heterogeneous physical systems at once, with a **shared embedding and normalization** across fields | One model matched or beat task-specific baselines without fine-tuning; fine-tuning beat training from scratch and fine-tuning video models on unseen physics |
| **DPOT**, Hao et al., 2024 | Auto-regressive **denoising** pretraining (inputs corrupted with noise) and Fourier attention; up to 0.5B parameters on 10+ PDE datasets with over 100,000 trajectories | Transferred to diverse downstream tasks, including 3D |
| **Poseidon**, Herde et al. (ETH Zürich), 2024 | A multiscale operator transformer with **time-conditioned layer norms** (continuous in time) and a **semigroup** training strategy that multiplies the training data from trajectories; pretrained on fluid dynamics | Outperformed baselines on 15 downstream tasks, generalized to physics not seen in pretraining, scaled with model and data size; open source |
| **Subramanian et al.** (LBNL / Berkeley), 2023 | A systematic study of scaling and transfer: model size, downstream data size, out-of-distribution physics parameters, mixtures of pretraining data | Fine-tuned pretrained models reached a target accuracy with orders of magnitude fewer downstream examples; gains grew with model size |

**What tends to transfer.**

- Generic **local dynamics** shared across PDEs (advection, diffusion, wave propagation) and multi-scale features.
- **Geometry understanding:** encoders that turn shapes into useful features transfer across tasks on the same kinds of geometry.
- **A good initialization and normalization scheme:** even when features are not reused, starting from a pretrained model can make optimization easier.

**What transfers poorly.**

- **New physics or new terms** (a new constitutive law, combustion, contact) that the corpus never contained.
- **New boundary-condition types and geometry classes** (internal flows when the corpus is external aerodynamics).
- **Regimes outside the corpus:** higher Reynolds or Mach numbers, transition, shocks; the code below shows a pretrained model that is excellent inside its parameter range and worse than training from scratch outside it.
- **New output fields** absent from pretraining, and inconsistent units or normalization.

**Central technical challenge: heterogeneity.** Different meshes, resolutions, variables, units, dimensions and time steps must fit one model: tokenization schemes, per-field normalization, conditioning on PDE parameters, and neural-field decoders are active research topics. See [Physics transformer architectures](physics-transformer-architectures.md) and the benchmark collection [The Well](public-simulation-datasets.md#pdebench-the-well-pdearena-cfdbench-and-apebench).

## Large geometry and large engineering models

The industrial version of the idea pretrains on **geometry and simulations of one engineering domain** rather than on many PDEs.

- **PhysicsX LGM-Aero (December 2024):** described as the first "Large Geometry Model" for aerospace, pretrained on more than **25 million meshes (over 10 billion vertices)** and **tens of thousands of CFD and FEA simulations** generated with Siemens tools. PhysicsX states that it infers aerodynamic performance, flight stability and structural stress for a broad class of flying shapes **zero-shot**, in under a second. The public **Ai.rplane** application, built on it, generates aircraft concepts and predicts their performance.
- **Luminary SHIFT models:** **SHIFT-SUV** (April 2025, with Honda and NVIDIA), trained on thousands of morphed variants of the open AeroSUV reference vehicle from FKFS with high-fidelity transient simulations, released with a public dataset (CC-BY-NC-4.0) and integrated into Blender for designers; **SHIFT-Wing** (2025) for transonic wing design. Luminary uses NVIDIA PhysicsNeMo's DoMINO architecture.
- **Emmi AI's AB-UPT** models were evaluated on DrivAerML and on the SHIFT-SUV and SHIFT-Wing data. See [AB-UPT](physics-transformer-architectures.md#ab-upt-anchor-tokens-and-divergence-free-fields).
- **NVIDIA** provides pretrained checkpoints (DoMINO, GeoTransolver and others on DrivAerML) and inference microservices through PhysicsNeMo.

**How to read vendor claims:** ask what "zero-shot" covers (shapes in the training distribution or genuinely new families), the test split, the metric (field error or integrated forces), the reference fidelity (RANS or scale-resolving), and how much fine-tuning a new customer needs. See [Evaluating generalization to unseen geometries](evaluation-metrics.md#evaluating-generalization-to-unseen-geometries).

## Lessons from ML weather forecasting

**What happened.**

- **FourCastNet** (NVIDIA, 2022), **Pangu-Weather** (Huawei, 2023) and **GraphCast** (DeepMind, 2023) learned global medium-range forecasting directly from the ERA5 reanalysis. GraphCast predicts hundreds of variables over 10 days at 0.25° resolution in under a minute and outperformed the leading operational deterministic system on 90% of 1,380 verification targets.
- **GenCast** (DeepMind, 2024) produced probabilistic ensemble forecasts with a diffusion model and beat ECMWF's operational ensemble on 97.4% of 1,320 targets. See [Probabilistic forecasting with diffusion](generative-models-for-physics.md#probabilistic-forecasting-with-diffusion).
- **ECMWF's AIFS** became operational on 25 February 2025, running alongside the physics-based IFS; ECMWF reports gains of up to 20% on some measures such as tropical-cyclone tracks and about 1,000 times less energy per forecast. **Aurora** (Microsoft, 2024) pretrained a large model on diverse atmospheric data and fine-tuned it for air quality, ocean waves and cyclone tracking.

**Why weather worked, and what it means for engineering.**

1. **Data:** decades of consistent, high-quality reanalysis (petabytes), on **one fixed geometry**, the Earth. Engineering has many geometries and far less data per family; data generation is the bottleneck.
2. **A strong numerical baseline existed** to learn from and compare against, with agreed metrics and benchmarks (WeatherBench). Engineering needs the same: shared datasets, metrics and leaderboards.
3. **Hybrid operation:** ML models run alongside physics models and are initialized from physics-based data assimilation; they did not replace the whole system.
4. **Probabilistic outputs** matter for chaotic systems.
5. **Pretrain-then-fine-tune works** when the downstream tasks share the same physics and geometry (Aurora).

## Fine-tuning on a small customer dataset

A customer has 50 simulations of their own product family and wants a model.

1. **Set up an honest evaluation first:** a grouped split by design (or design family), fixed before any experiment, with a held-out test set touched once. See [Splitting simulation data](data-splits-and-leakage.md#splitting-simulation-data).
2. **Establish baselines:** a GP or POD + GP on design parameters, a nearest-neighbour model, and a small model trained from scratch. A foundation model must beat them.
3. **Check zero-shot performance** of the pretrained model; if the customer's designs are inside its training distribution, it may already be good, and fine-tuning on a few cases can make it worse.
4. **Fine-tune conservatively:** a small learning rate, few steps, early stopping on validation; freeze the encoder and train the head, or use parameter-efficient adapters (LoRA); align units and normalization with the pretraining data.
5. **Watch for negative transfer:** if the pretraining physics differs (another regime, other boundary conditions), the pretrained weights can be worse than random initialization, as the code below shows.
6. **Check calibration and out-of-distribution behaviour** on the customer's data, not just accuracy.
7. **Report the gain over baselines honestly,** with seeds and confidence intervals. If there is none, say so.

The system-design version is [Design: a fine-tuning pipeline](design-fine-tuning-pipeline.md).

## Is data the moat?

**Arguments for:**

- High-fidelity simulation data is expensive (thousands of core-hours per case for scale-resolving CFD), slow to produce and needs expertise to produce well.
- Physics does not go out of date: a well-made dataset keeps its value for years.
- Proprietary data from real products (with test validation) is more representative than academic geometries.

**Arguments against:**

- **Open datasets keep growing:** DrivAerML, DrivAerNet++, AhmedML, WindsorML, SHIFT-SUV, The Well, and more each year. See [Public simulation datasets](public-simulation-datasets.md).
- **Customer data belongs to the customer** and often cannot be pooled across customers.
- **Data can be generated** by anyone with compute and a solver; cost falls as GPU solvers improve.
- **Other moats may matter as much:** integration into engineering workflows, validation and trust with conservative engineering organizations, the ability to deliver results (forward-deployed engineering), and the efficiency of data generation itself (multi-fidelity, active learning).

**Licences matter:** many open datasets are non-commercial (CC BY-NC), which limits their use in commercial models. A balanced interview answer: data is necessary but not sufficient; the durable advantage is the combination of data, validated models and integration into how customers work.

## Code: when pretraining helps, and when it hurts

A small operator learns to map a forcing function to the solution of −u″ + c u³ = f (a nonlinear steady problem solved with Newton's method). It is pretrained on 2,100 solved problems with nonlinearity strengths c between 0 and 800. A "customer" then brings 20 solved cases of a new problem, either inside the pretraining range (c = 400) or outside it (c = 1,600). The comparison: training from scratch on the 20 cases, the pretrained model zero-shot, the pretrained model fine-tuned on the 20 cases, and a model pretrained on mismatched physics then fine-tuned. Training takes under a minute.

```python
import copy
import numpy as np
import torch, torch.nn as nn

rng = np.random.default_rng(0)
n = 64
x = np.linspace(0, 1, n + 2)[1:-1]
h = x[1] - x[0]
D2 = (np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) / h ** 2

def solve(f, c, k=1.0):
    """'The simulator': -k u'' + c u^3 = f on (0, 1), u = 0 at both ends, finite differences and Newton's method."""
    u = np.zeros(n)
    for _ in range(30):
        du = np.linalg.solve(-k * D2 + np.diag(3 * c * u ** 2), -(-k * D2 @ u + c * u ** 3 - f))
        u += du
        if np.abs(du).max() < 1e-12:
            break
    return u

def make_task(m, c, k=1.0):
    a = rng.normal(size=(m, 6)) / np.arange(1, 7)
    F = 30 * a @ np.sin(np.outer(np.arange(1, 7), np.pi * x))            # random smooth forcing functions
    U = np.stack([solve(f, c, k) for f in F])
    return torch.tensor(F, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

class Operator(nn.Module):
    """Forcing (64 values) and the nonlinearity strength c -> solution (64 values)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.head = nn.Linear(256, n)
    def forward(self, f, c):
        return 0.1 * self.head(self.encoder(torch.cat([f / 30, c], 1)))

def train(model, F, U, C, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(F), (min(64, len(F)),))
        loss = ((model(F[idx], C[idx]) - U[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

def rel_err(model, F, U, C):
    with torch.no_grad():
        return ((model(F, C) - U).norm(dim=1) / U.norm(dim=1)).mean().item()

# Pretraining corpus: 2,100 solved problems spanning nonlinearity strengths c = 0 ... 800.
cs = [0.0, 100.0, 200.0, 300.0, 500.0, 600.0, 800.0]
corpus = [make_task(300, c) for c in cs]
F_pre, U_pre = torch.cat([t[0] for t in corpus]), torch.cat([t[1] for t in corpus])
C_pre = torch.cat([torch.full((300, 1), c / 800) for c in cs])
torch.manual_seed(0); pretrained = train(Operator(), F_pre, U_pre, C_pre, 8000, 1e-3)
# A mismatched corpus: different physics (linear, with a much smaller diffusion coefficient).
F_bad, U_bad = make_task(2100, 0.0, k=0.3)
torch.manual_seed(0); mismatched = train(Operator(), F_bad, U_bad, torch.zeros(2100, 1), 8000, 1e-3)

for c_customer in [400.0, 1600.0]:                                    # inside and outside the pretraining range
    F_c, U_c = make_task(20, c_customer)                              # the customer's 20 simulations
    F_t, U_t = make_task(500, c_customer)                             # held-out test cases
    C_c, C_t = torch.full((20, 1), c_customer / 800), torch.full((500, 1), c_customer / 800)
    torch.manual_seed(0)
    rows = {"from scratch on 20 cases": train(Operator(), F_c, U_c, C_c, 2000, 1e-3),
            "pretrained, zero-shot": pretrained,
            "pretrained + fine-tuned on 20 cases": train(copy.deepcopy(pretrained), F_c, U_c, C_c, 1000, 2e-4),
            "mismatched pretraining + fine-tuned": train(copy.deepcopy(mismatched), F_c, U_c, C_c, 1000, 2e-4)}
    print(f"customer problem c = {c_customer:.0f} ({'inside' if c_customer <= 800 else 'outside'} the pretraining range):")
    for name, model in rows.items():
        print(f"   {name:<38} relative L2 error {rel_err(model, F_t, U_t, C_t):.3f}")
```

Output:

```
customer problem c = 400 (inside the pretraining range):
   from scratch on 20 cases               relative L2 error 0.246
   pretrained, zero-shot                  relative L2 error 0.063
   pretrained + fine-tuned on 20 cases    relative L2 error 0.095
   mismatched pretraining + fine-tuned    relative L2 error 1.832
customer problem c = 1600 (outside the pretraining range):
   from scratch on 20 cases               relative L2 error 0.283
   pretrained, zero-shot                  relative L2 error 0.373
   pretrained + fine-tuned on 20 cases    relative L2 error 0.211
   mismatched pretraining + fine-tuned    relative L2 error 2.432
```

**What it shows.**

- **Inside the pretraining range,** the pretrained model zero-shot is four times more accurate than a model trained from scratch on the customer's 20 cases (6% against 25% error), and fine-tuning on those 20 cases makes it **worse** (9.5%): with so little data, fine-tuning overfits and forgets. Always compare against zero-shot.
- **Outside the range,** zero-shot fails (37%, worse than from scratch), and fine-tuning helps: 21% against 28% from scratch. The pretraining still provides a useful starting point.
- **Mismatched pretraining** is far worse than starting from scratch in both cases: the learned features encode the wrong physics and scale, and 20 cases cannot undo them. This is negative transfer.

## In an interview

**Thirty-second answer to "What is a PDE foundation model, and what transfers?"** "A large model pretrained on many simulations, often many PDE families, so that it can be used zero-shot or fine-tuned with little data. MPP, DPOT and Poseidon show that pretraining can cut the downstream data needed by large factors, more so for bigger models. What transfers is shared local physics, geometry encoders and good initializations; what does not is new physics, new boundary-condition types, regimes outside the corpus and new output fields. In practice I would always compare zero-shot, fine-tuned and from-scratch models and simple baselines on a grouped split, because fine-tuning on 20 cases can help, do nothing, or hurt."

**Likely follow-ups**

- *What did weather forecasting teach us?* Learned models can beat numerical ones when there is abundant consistent data, a strong baseline and agreed benchmarks; engineering lacks the first and third.
- *Is data the moat?* Necessary but not sufficient; integration, validation and trust matter as much, and licences constrain open data.
- *How do you avoid negative transfer?* Check zero-shot and from-scratch baselines, align normalization, fine-tune conservatively with early stopping, and prefer pretraining data close to the target physics.

## References

- [Multiple Physics Pretraining for Physical Surrogate Models (McCabe et al.)](https://arxiv.org/abs/2310.02994)
- [DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (Hao et al.)](https://arxiv.org/abs/2403.03542)
- [Poseidon: Efficient Foundation Models for PDEs (Herde et al.)](https://arxiv.org/abs/2405.19101)
- [Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior (Subramanian et al.)](https://arxiv.org/abs/2306.00258)
- [PhysicsX: Introducing LGM-Aero and Ai.rplane](https://www.physicsx.ai/newsroom/introducing-lgm-aero-genai-for-aero-engineering-and-airplane-showcase-application-for-aerostructures)
- [Luminary: Introducing SHIFT models](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/) and the [SHIFT-SUV dataset](https://huggingface.co/datasets/luminary-shift/SUV)
- [GraphCast (Lam et al.)](https://arxiv.org/abs/2212.12794), [GenCast (Price et al.)](https://arxiv.org/abs/2312.15796), [Aurora (Bodnar et al.)](https://arxiv.org/abs/2405.13063), [FourCastNet (Pathak et al.)](https://arxiv.org/abs/2202.11214)
- [ECMWF: AI forecasts become operational (25 February 2025)](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational)
