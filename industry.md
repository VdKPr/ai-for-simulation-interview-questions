# Industry and Interview Notes

[← Questions](README.md) · [All answers](answers/README.md)

*Facts on this page were checked in October 2026 against the linked sources. Funding, products and job descriptions change quickly, so check them again before an interview.*

**Short answer.** The field has three kinds of players:

- **Start-ups built on physics AI:** PhysicsX, Neural Concept, Luminary, Emmi AI, BeyondMath, Navier.
- **Large CAE vendors adding AI to their suites:** Ansys (now part of Synopsys) with SimAI; Siemens, with Altair's physicsAI and its own Simcenter.
- **Platform and compute companies:** NVIDIA with PhysicsNeMo; CoreWeave, which bought Monolith; Rescale.

They differ in what they sell (a platform, pretrained models, engineers who deliver a use case, or compute), in which customers they target (aerospace and automotive first), and in how they handle the trust problem. Interviewers expect you to know their product, their papers and their customers, and to have a view on why customers still hesitate.

**On this page**

- [Companies](#companies)
- [Ansys SimAI](#ansys-simai)
- [NVIDIA PhysicsNeMo](#nvidia-physicsnemo)
- [Synopsys-Ansys and Siemens-Altair](#synopsys-ansys-and-siemens-altair)
- [PhysicsX LGM-Aero and Ai.rplane](#physicsx-lgm-aero-and-airplane)
- [Luminary SHIFT models and datasets](#luminary-shift-models-and-datasets)
- [How physics-AI companies make money](#how-physics-ai-companies-make-money)
- [Which industries buy, and why customers hesitate](#which-industries-buy-and-why-customers-hesitate)
- [What the job descriptions ask for](#what-the-job-descriptions-ask-for)
- [Reported interview formats](#reported-interview-formats)
- [In an interview](#in-an-interview)

## Companies

| Company | What they build | Public signals | Links |
| --- | --- | --- | --- |
| **PhysicsX** (London) | A physics-AI platform embedded in engineering workflows, pretrained "Large Physics Models", and engineers who deliver use cases with customers | LGM-Aero and the Ai.rplane demo (Dec 2024); $135M Series B (June 2025); $300M Series C at about $2.4B, led by Temasek, with NVIDIA, Siemens and Applied Materials among existing investors (8 June 2026). The company reported more than 300 staff and doubled recognized revenue year on year | [Site](https://www.physicsx.ai/) · [Series C](https://www.physicsx.ai/newsroom/physicsx-announces-300m-series-c-to-accelerate-physics-ai-for-industrial-engineering) · [Series B](https://www.physicsx.ai/newsroom/physicsx-raises-135m-series-b-to-usher-in-a-new-era-of-ai-native-engineering-and-manufacturing) |
| **Neural Concept** (Lausanne) | An AI-native engineering design platform built on geometric deep learning on 3D shapes | EPFL Computer Vision Lab spin-off (founded 2018 by Pierre Baqué, with François Fleuret and Pascal Fua); $100M Series C (Dec 2025); a forward-deployed engineer role in Bengaluru | [Site](https://www.neuralconcept.com/) · [Series C](https://tech.eu/2025/12/18/neural-concept-raises-100m-to-scale-ai-native-engineering/) · [Jobs](https://www.neuralconcept.com/jobs) |
| **Luminary** (San Mateo) | Cloud CFD on GPUs plus physics-AI models trained on its own simulations | SHIFT-SUV with Honda and NVIDIA (Apr 2025) and SHIFT-Wing with Otto Aviation (Jul 2025), both released as datasets | [Site](https://www.luminarycloud.com/) · [SHIFT](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/) |
| **Emmi AI** (Linz) | "Large Engineering Models" for real-time simulation | €15M seed (Apr 2025); the AB-UPT papers; the open Noether framework | [Site](https://www.emmi.ai/) · [Seed](https://www.emmi.ai/news/emmi-ai-raises-eur-15m-to-bring-ai-to-the-heart-of-industrial-engineering) · [AB-UPT](https://arxiv.org/abs/2502.09692) · [Noether](https://github.com/Emmi-AI/noether) |
| **Monolith** (London; part of CoreWeave) | No-code ML for engineering test data: anomaly detection, test-plan optimization, next-test recommendation | Imperial College spin-off (2016); CoreWeave announced the acquisition on 6 October 2025, alongside its purchases of Weights & Biases and OpenPipe | [Site](https://www.monolithai.com/) · [Acquisition](https://www.coreweave.com/news/coreweave-to-acquire-monolith-expanding-ai-cloud-platform-into-industrial-innovation) |
| **Navier** (San Francisco) | AI agents that automate work between CAD, simulation and validation ("agent-driven engineering") | $5.6M seed from GV, HCVC and Y Combinator, launched from stealth (Dec 2025) | [Site](https://www.navier.ai/) · [Seed](https://www.businesswire.com/news/home/20251215832267/en/Navier-Raises-%245.6M-to-Deploy-Agent-Driven-Engineering-the-Next-Engineering-Unlock-After-CAD-and-Simulation) |
| **BeyondMath** | A "generative physics engine" | Foundation model for physics | [Site](https://www.beyondmath.com/) |
| **NVIDIA** | PhysicsNeMo (open source) and the GPUs underneath most of the field | See below | [PhysicsNeMo](https://developer.nvidia.com/physicsnemo) |
| **Ansys, part of Synopsys** | SimAI, GeomAI and the first agentic features in the Ansys suite | See below | [SimAI](https://www.ansys.com/products/simai) |
| **Siemens** | Simcenter, plus Altair's physicsAI and HyperWorks after the acquisition | See below | [Simcenter](https://www.siemens.com/global/en/products/software/simcenter.html) |
| **Rescale** | Cloud HPC and simulation platform positioned around "AI physics" | | [Site](https://www.rescale.com/) |

**How they differ, in one line each.** PhysicsX sells deep, use-case delivery with its own pretrained models; Neural Concept sells a design platform used by engineering teams; Luminary uses its own GPU CFD to generate the data its models need; Emmi AI is research-led around one architecture family; Monolith works from **test** data rather than simulation; Navier automates the workflow around solvers rather than replacing them; the CAE vendors add AI where their customers already work.

## Ansys SimAI

- **What it is:** a software-as-a-service platform, launched on 9 January 2024, where customers upload their own previous 3D simulation results (from Ansys or other solvers), train a model, and predict fields for new designs in minutes.
- **How it works, as Ansys describes it:**
  - It is **physics-agnostic**: it learns from solutions, not equations, and covers fluids, structures, electromagnetics and optics.
  - It takes the **shape itself** as input, so the designs do not need a common parametrization.
  - It uses an **implicit neural representation**, so fields can be queried at any point and resolution, and global coefficients are computed from the predicted fields.
  - Its **confidence score** measures the distance from the nearest known geometry in a learned high-dimensional space, which is a form of OOD detection. See [OOD detection](answers/out-of-distribution-and-generalization.md).
- **2026 R1 (March 2026):** SimAI now comes in two tiers. **SimAI Premium** is the cloud SaaS product; **SimAI Pro** runs on the desktop for projects that need data stored locally, which answers the "we will not upload our CAD" objection. optiSLang connectors cover the whole loop (generate training data, train, optimize), and PySimAI scripts it from Python.
- **Why it matters in an interview:** it is the clearest example of a vendor building a surrogate on the customer's own archived data, and of the design choices this guide discusses: non-parametric geometry input, query-anywhere decoders, and a built-in trust score.

## NVIDIA PhysicsNeMo

- **What it is:** NVIDIA's open-source (Apache 2.0) framework for physics-ML, formerly called Modulus. It provides models, training utilities, data pipelines and reference recipes.
- **Models included:** MeshGraphNet and X-MeshGraphNet; DoMINO and GeoTransolver for external aerodynamics; Fourier and other neural operators; PINN utilities; diffusion models such as CorrDiff for weather downscaling.
- **Recipes:** external aerodynamics on DrivAerML, crash, semiconductor packaging and data-center thermal design, among others.
- **Why it matters:** many start-ups and customers start from it, and its papers (DoMINO, X-MeshGraphNet, GeoTransolver) are benchmarks you may be asked to compare against. NVIDIA invests in several of the start-ups (PhysicsX among them) and partners with others (Luminary on SHIFT-SUV).
- See [Physics transformer architectures](answers/physics-transformer-architectures.md) and [Scaling to large meshes](answers/scaling-to-large-meshes.md).

## Synopsys-Ansys and Siemens-Altair

- **Synopsys bought Ansys.** The deal was announced in January 2024 (about $35B) and completed on 17 July 2025. The logic is "silicon to systems": chips, packages and the products they go into are designed and simulated together.
- **What it produced so far:** Ansys 2026 R1 (11 March 2026) was the first release with joint Synopsys-Ansys workflows. It added:
  - **GeomAI**, generative concept design learned from reference designs;
  - **SimAI Pro**, the desktop tier of SimAI;
  - the suite's first agentic features: **Mesh Agent** in Mechanical (exploratory use) to debug meshing failures, and the **Discovery Validation Agent** (in early customer evaluation) to flag set-up issues;
  - **Engineering Copilot** in more products.
- **Siemens bought Altair.** The deal completed in March 2025 (about $10B enterprise value), bringing Altair's HyperWorks, its physicsAI tool and its data-analytics products into Siemens's Xcelerator portfolio alongside Simcenter.
- **What it means:** the two largest CAE ecosystems now own AI products and sell them inside existing licences. Start-ups compete by going deeper on specific use cases, by being solver-agnostic, and by pretrained models. Siemens is also an investor in PhysicsX and generated the training simulations behind LGM-Aero.

## PhysicsX LGM-Aero and Ai.rplane

- **LGM-Aero** (4 December 2024) is a "Large Geometry Model" for aerospace: a model of geometry and physics pretrained on **more than 25 million meshes** (over 10 billion vertices) and **tens of thousands of CFD and FEA simulations** generated with Siemens tools, trained on AWS.
- **What it does:** it generates new shapes and, zero-shot, predicts lift, drag, flight stability and structural stress for a broad class of flying shapes, in under a second rather than hours.
- **Ai.rplane** is a free public demo built on it: you explore a 2D map of a learned design space (each design has a 512-number latent "signature", compressed to two dimensions), generate new aircraft, and see predicted performance with uncertainty.
- **PhysicsX's own caveat** on the demo page: if a geometry is "a bit rogue", the predictions will not be good; for shapes that resemble aircraft, they are typically comparable to simulation. That is an honest statement of the interpolation-vs-extrapolation issue in [Out-of-distribution and generalization](answers/out-of-distribution-and-generalization.md).
- **Why it matters:** it shows the foundation-model strategy: pretrain broadly on geometry, then fine-tune on customer data. See [Foundation models for physics](answers/foundation-models-for-physics.md).

## Luminary SHIFT models and datasets

- **SHIFT-SUV** (April 2025, with Honda and NVIDIA): thousands of morphed variants of the AeroSUV body, simulated with Luminary's GPU CFD, with a pretrained model.
- **SHIFT-Wing** (July 2025, with Otto Aviation): transonic wing variants.
- **Release model:** the datasets are on Hugging Face under CC-BY-NC-4.0 (non-commercial), and commercial licences are available. They are used as benchmarks in AB-UPT and GeoTransolver papers.
- **Why it matters:** it shows how a company that owns a fast solver turns compute into data, data into models, and both into something to sell. Check the licence before using these datasets in commercial work.
- Links: [SHIFT announcement](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/) · [SHIFT-Wing](https://luminary.ai/resources/luminary-cloud-releases-shift-wing-a-physics-ai-model-for-transonic-wing-design-in-collaboration-with-otto-aviation-leveraging-nvidia-technologies/) · [SHIFT-SUV dataset](https://huggingface.co/datasets/luminary-shift/SUV)

## How physics-AI companies make money

| Model | What the customer pays for | Who does it | Trade-off |
| --- | --- | --- | --- |
| **Platform licence** | Seats or usage to train and run models on their own data (cloud or on-premises) | Neural Concept, SimAI, Siemens physicsAI | Scales well; the customer needs data and skills |
| **Forward-deployed engineering and services** | Engineers working inside the customer's team to deliver a use case end to end | PhysicsX, Navier, Luminary, Neural Concept roles | High value and trust; less scalable, so usually combined with a platform |
| **Pretrained models and datasets** | Commercial licences for models or data released openly for research | Luminary SHIFT, PhysicsX LGM models | Differentiates; data generation is expensive |
| **Bundled in CAE suites** | Part of existing simulation licences | Ansys (Synopsys), Siemens | Easy to buy; tied to one vendor's tools |
| **Compute** | GPUs and the platform on top | NVIDIA, CoreWeave (with Monolith), Rescale | Profits whoever wins at the model layer |

Most start-ups combine a platform with forward-deployed engineering: the engineers win the first use case and the platform keeps it running. That is why so many roles in [the job descriptions below](#what-the-job-descriptions-ask-for) are customer-facing.

## Which industries buy, and why customers hesitate

**Who buys.** Industries where each simulation is expensive, design iterations are many, and a better product is worth a lot:

- **Aerospace:** aerodynamics, aerostructures, engines (LGM-Aero, SHIFT-Wing).
- **Automotive:** external aerodynamics, crash, thermal management, batteries (SHIFT-SUV, DrivAerML).
- **Semiconductors and electronics:** packaging, thermal design, warpage (Synopsys-Ansys, PhysicsNeMo recipes; Applied Materials invests in PhysicsX).
- **Energy and industrial equipment:** turbomachinery, heat exchangers, data-center cooling.
- **Test-heavy industries** (batteries, motorsport): ML on test data rather than simulation (Monolith).

**Why customers hesitate.**

| Concern | What lies behind it | How a good team answers |
| --- | --- | --- |
| **Trust** | No error guarantee; silent failure on new designs | Validation on the customer's held-out cases, uncertainty and OOD gates, solver fallback ([customer scenarios](answers/customer-scenarios.md#i-dont-trust-black-boxes)) |
| **Data** | Too few simulations; old data in mixed formats; the important regimes missing | Pretrained models, multi-fidelity, active learning, a data pipeline ([design page](answers/design-simulation-data-pipeline.md)) |
| **IP and security** | Reluctance to upload CAD to someone else's cloud | On-premises or private-cloud deployment, desktop tiers (SimAI Pro), models trained where the data lives |
| **Integration** | It must fit existing CAD, CAE and sign-off processes | Plug-ins and APIs into existing tools; outputs in the formats engineers already use |
| **Accountability** | Who signs off on a number that came from a surrogate? | Use surrogates for exploration and ranking; verify final designs with the solver |
| **Return on investment** | Generating training data costs solver hours | Show the break-even number of queries; reuse archived data |

## What the job descriptions ask for

| Role | Key requirements (paraphrased from the posting) | Source |
| --- | --- | --- |
| PhysicsX: Machine Learning Engineer (NYC / SF) | ML on 3D graphs and point clouds for real engineering; data pipelines; Python ML pipelines; Spark/Dask; cloud and HPC; Docker/Kubernetes; versioning, testing, CI/CD, API design, MLOps; customer-facing project delivery; physics or engineering background; travel to customer sites about 3–4 weeks per quarter | [NYC posting](https://job-boards.greenhouse.io/physicsx/jobs/4849382101) · [SF posting](https://job-boards.greenhouse.io/physicsx/jobs/4880947101) |
| Neural Concept: Forward Deployed Engineer (Bengaluru) | Master's or PhD in engineering, applied maths or physics; strong engineering *and* ML; strong Python and ML/DL frameworks; simulation experience (CAD/CAE, CFD, FEM a plus); customer-facing communication; fluent English | [Posting](https://builtinbengaluru.in/job/application-engineer/8741962) |
| Luminary: Senior/Principal Forward Deployed Engineer, Applied AI/ML | Model development for physics-AI engagements; validation against ground-truth simulation; neural operators, diffusion models, geometric deep learning; working knowledge of CFD/FEA/EM/thermal; distributed training, experiment tracking, model registries, inference serving; "rigorous experimentation and honest evaluation" | [Posting](https://ats.rippling.com/luminarycloud/jobs/c2b9847d-186c-4675-a32f-1a9ef7d478b4) |
| NVIDIA: Senior Physics Machine Learning Engineer, CAE (posting closed) | Strong Python; graph networks, diffusion models and RL; PyTorch/TensorFlow/JAX; ML for engineering simulation (CFD, structural, thermal, EM, optics, acoustics); a plus: scientific visualization, multi-node data/model parallelism, CUDA, simulation codes | [Posting](https://hitmarker.net/jobs/nvidia-senior-physics-machine-learning-engineer-cae-1890456) |

## Reported interview formats

Self-reported and incomplete; formats differ by team and seniority.

- **PhysicsX** (Glassdoor; four reviews shown when checked):
  - HR call → coding assessment (three problems on Coderbyte) → role-fit technical discussion → technical task simulating an engineering use case → presentation of a technical project.
  - Reported questions include motivation, *how uncertainty is predicted in Gaussian processes*, and general data preprocessing.
  - A 2026 New York report mentions a coding question about converting flat vertex data into a mesh format, and live coding on a mesh dataset (identify the elements, find the bounding box). A 2026 London report mentions a 30-minute data-structures round, a technical introduction, a 1–1.5 hour pair-programming session and a final round.
  - [Glassdoor](https://www.glassdoor.com.mx/Entrevista/PhysicsX-Entrevista-E9306867-RVW97800229.htm)
- **Neural Concept:** a public summary lists a coding test, a culture interview and a technical round; the details are behind a login. [1Point3Acres](https://www.1point3acres.com/interview/thread/1156027)
- **Aggregator guides** (useful for structure, but treat them as indicative; they mix sources): [InterviewQuery: PhysicsX ML Engineer](https://www.interviewquery.com/interview-guides/physicsx-ml-engineer) · [Dataford: PhysicsX ML Engineer](https://dataford.io/interview-guides/physicsx/machine-learning-engineer)

## Applying from India

Neural Concept advertises a forward-deployed role in Bengaluru (above). Large CAE vendors and NVIDIA also hire engineers in India; filter their careers pages by location and search for "physics AI", "simulation" and "CAE".

## In an interview

**Thirty-second answer to "What do you know about us and our competitors?"** Name the company's product, its latest model or paper and one customer story; then place it: "You sell a platform plus forward-deployed delivery with your own pretrained models; Neural Concept sells a design platform; Luminary owns a fast GPU solver and turns it into data and models; and the CAE vendors, Ansys inside Synopsys and Siemens with Altair, now bundle AI into suites customers already pay for. Your advantage is depth on hard use cases and models that are not tied to one solver; the risk is that bundled tools become good enough. The barrier for everyone is trust, which is why validation and uncertainty matter as much as the architecture."

**Likely follow-ups**

- *Why would a customer pick a start-up over the tool bundled in their CAE licence?* Depth on their use case, solver independence, better models on free-form geometry, and engineers who deliver.
- *What is the biggest risk to this business?* Data cost and trust; plus incumbents bundling "good enough" AI.
- *How would you handle a customer who will not upload CAD?* On-premises or private-cloud deployment, training where the data lives, or models that run on the desktop.

## References

- [PhysicsX Series C](https://www.physicsx.ai/newsroom/physicsx-announces-300m-series-c-to-accelerate-physics-ai-for-industrial-engineering) and [LGM-Aero announcement](https://www.physicsx.ai/newsroom/introducing-lgm-aero-genai-for-aero-engineering-and-airplane-showcase-application-for-aerostructures); [Welcome to Ai.rplane](https://www.physicsx.ai/newsroom/welcome-to-airplane)
- [Ansys launches SimAI (Jan 2024)](https://www.ansys.com/news-center/press-releases/1-9-24-ansys-launches-simai), [Explaining SimAI](https://ansys.synopsys.com/blog/explaining-simai) and [Synopsys launches Ansys 2026 R1](https://news.synopsys.com/2026-03-11-Synopsys-Launches-Ansys-2026-R1-to-Re-Engineer-Engineering-with-Joint-Solutions-and-AI-Powered-Products)
- [Synopsys completes acquisition of Ansys](https://news.synopsys.com/2025-07-17-Synopsys-Completes-Acquisition-of-Ansys) and [Siemens completes acquisition of Altair](https://www.digitalengineering247.com/article/siemens-completes-acquisition-of-altair)
- [CoreWeave to acquire Monolith](https://www.coreweave.com/news/coreweave-to-acquire-monolith-expanding-ai-cloud-platform-into-industrial-innovation); [Navier seed round](https://www.businesswire.com/news/home/20251215832267/en/Navier-Raises-%245.6M-to-Deploy-Agent-Driven-Engineering-the-Next-Engineering-Unlock-After-CAD-and-Simulation)
- [NVIDIA PhysicsNeMo on GitHub](https://github.com/NVIDIA/physicsnemo) and [documentation](https://docs.nvidia.com/physicsnemo/latest/)
