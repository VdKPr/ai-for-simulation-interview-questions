# Simulation Agents: The Research

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Between 2023 and 2026 a wave of papers built LLM agents that set up and run engineering simulations. **FEABench** (Google, 2025) is the reference benchmark for finite-element reasoning: agents operate COMSOL Multiphysics through its API on 15 manually verified problems; the best agent produced executable code 88% of the time but solved no problem completely and correctly. **CFD agents** for OpenFOAM matured fastest because OpenFOAM is open source with many public tutorials: **MetaOpenFOAM** (role-based multi-agent pipeline with retrieval, 85% pass rate on 8 tasks), **OpenFOAMGPT** (retrieval and iterative correction with GPT-4o and o1, with a warning that human oversight is still needed), and **Foam-Agent** (stage-specific multi-index retrieval, dependency-aware file generation and a reviewer loop; 88.2% success on 110 tasks against 55.5% for MetaOpenFOAM). **FEA agents** span free code generation (MechAgents with FEniCS), input-deck generation (FeaGPT with CalculiX, MooseAgent with MOOSE), harnesses around commercial solvers (CAX-Agent for Ansys MAPDL, with a recovery ladder that raised completion from 69% to 93%), and constrained interfaces where the LLM only fills a specification for validated templates (Penn State's FEniCS system). The common lessons: structure and recovery loops matter more than clever prompting, executable is not correct, and verification is the unsolved part.

**On this page**

- [FEABench](#feabench)
- [CFD agents: Foam-Agent, MetaOpenFOAM and OpenFOAMGPT](#cfd-agents-foam-agent-metaopenfoam-and-openfoamgpt)
- [FEA agents](#fea-agents)
- [Lessons across systems](#lessons-across-systems)
- [Commercial agentic features](#commercial-agentic-features)
- [In an interview](#in-an-interview)

## FEABench

**What it is.** A benchmark (Mudur, Cui, Venugopalan, Raccuglia, Brenner and Norgaard; Google and Harvard; NeurIPS 2024 workshops, arXiv 2025) for LLMs and agents that solve physics, mathematics and engineering problems with finite-element analysis by operating **COMSOL Multiphysics** through its Java API.

**Design.**

- **FEABench Gold:** 15 problems, each manually verified to be self-contained, solvable, and quantitatively checkable: if the model is built correctly, a target value is computed and exported.
- **FEABench Large:** 200 algorithmically parsed problems for broader, less strict evaluation.
- Two task formats: **ModelSpecs** (a natural problem description) and **Plan** (step-by-step modelling instructions).
- **Evaluation:** executability of API calls and a set of intermediate metrics measuring "distance to a correct solution" (geometry, physics interfaces and features, boundary conditions), plus whether a valid target is computed within 10% of the answer.
- **An agent** with tools: execute code and read errors, look up physics-interface documentation, retrieve annotated code examples, and iterate (40 candidate solutions per problem: 20 sampled, 20 corrections).

**Findings.**

- Interaction helps: executability rose from 0.62 to 0.88 with the multi-turn agent, the best of all strategies.
- The **physics block** of the code (physics interfaces, boundary conditions) was the hardest part to make executable.
- **Correctness lags far behind executability:** only one problem produced a value within 10% of the answer, and it was an accident: the exported 20 °C was COMSOL's default temperature, against a correct 18.3 °C. The authors conclude that the benchmark is far from solved.
- In a side experiment, a software-engineering agent writing plain Python computed a valid target for 11 problems but met the 10% criterion for only 4: without a verified physics library, the agent must write the equations itself, and precision becomes the bottleneck.

**Why it matters.** It is the cleanest evidence for the main design rule of simulation agents: measure physics correctness, not just whether the code runs, and do not let the model invent the physics. See [Why the LLM should not write solver physics](llm-agents-for-simulation.md#why-the-llm-should-not-write-solver-physics).

## CFD agents: Foam-Agent, MetaOpenFOAM and OpenFOAMGPT

An OpenFOAM case is a directory of interdependent text dictionaries (mesh in `constant/polyMesh`, boundary and initial conditions in `0/`, solver controls in `system/controlDict`, `fvSchemes`, `fvSolution`), so automating CFD means generating a consistent set of files, running mesh generation and the solver, and interpreting logs.

| | MetaOpenFOAM (2024) | OpenFOAMGPT (2025) | Foam-Agent / Foam-Agent 2.0 (2025) |
| --- | --- | --- | --- |
| Core idea | MetaGPT-style assembly line of roles (architect, input writer, runner, reviewer) with LangChain retrieval over OpenFOAM tutorials | Retrieval-augmented generation with GPT-4o and o1-preview, plus an iterative correction loop | Six specialist agents (planning, meshing, file writing, execution, review, visualization) |
| Retrieval | Tutorials as a single knowledge base | RAG over cases and documentation | **Multi-index retrieval** along four structural dimensions, selected by workflow stage |
| File generation | Role-based | LLM generation with correction | **Dependency-aware**: files generated in topological order of the case dependency graph, each in the context of its predecessors |
| Error handling | Reviewer role | Iterative correction from error messages | **Trajectory-conditioned reviewer loop**: each repair conditioned on the history of previous attempts and diagnoses, with minimal edits |
| Extras | Natural-language interface | Comparison of model cost and capability (o1 about 6 times the token price, better on complex tasks) | Meshing agent (external meshes, gmsh), HPC submission scripts, ParaView post-processing, tools exposed through the Model Context Protocol |
| Reported result | 85% pass rate on a benchmark of 8 CFD tasks, about $0.22 per test | Successful automation of a range of cases; human oversight remains crucial; performance fluctuates over model versions, so monitor it | 88.2% success on 110 simulation tasks with Claude 3.5 Sonnet, against 55.5% for MetaOpenFOAM |

**What to take from them.**

1. **Retrieval organized around the workflow** (which file, which stage) beats one generic index.
2. **Cross-file consistency** is a real failure mode; generating files in dependency order addresses it.
3. **Repair loops do most of the work:** errors are informative, and conditioning repairs on previous attempts avoids loops of the same mistake.
4. **Success means "ran to completion and passed checks"**, usually on tutorial-like cases; physical accuracy against references, unusual geometries and turbulence-model choices remain open. Foam-Agent's authors note that execution success does not imply physical fidelity.

## FEA agents

| System | Solver | Approach | Reported results |
| --- | --- | --- | --- |
| **MechAgents** (Ni and Buehler, MIT, 2023) | FEniCS | A two-agent team writes, executes and self-corrects code for elasticity problems (various boundary conditions, geometries, meshes, small and finite deformation, linear and hyperelastic materials); a larger team splits planning, formulation, coding, execution and criticism | Agents correct each other's errors (including wrong weak forms and traction formulas) and improve team performance |
| **FeaGPT** (Qi, Xu and Chu, 2025) | CalculiX (with geometry and meshing tools) | A geometry-mesh-simulation-analysis (GMSA) pipeline from a conversational description, including adaptive meshing and boundary-condition inference | Turbocharger cases (a 7-blade compressor, a 12-blade turbine at 110,000 rpm) and a 432-configuration NACA airfoil study |
| **MooseAgent** (2025) | MOOSE | Generates MOOSE input files from natural language with a vector database of annotated MOOSE documentation, orchestrated with LangGraph | 93% average success; under 1 yuan per case |
| **CAX-Agent** (2026) | Ansys MAPDL | A harness with three layers (LLM service, agent harness, solver backend) and a **recovery ladder**: deterministic rule patching, then model-driven regeneration, then context enrichment, then a human | 50 structural benchmarks × 3 runs: completion 0.693 with no recovery, 0.773 with rules only, 0.927 with model-driven recovery; task score 3.59 of 4; two blind raters agreed with a quadratic weighted kappa of 0.84 |
| **Constrained FEniCS interface** (Upadhyay and Reinhart, Penn State, 2026) | FEniCS | The LLM only parses the prompt into a JSON specification (and writes gmsh code for geometry outside a catalogue); a deterministic dispatcher calls 5 human-written FEniCS templates; the LLM never writes weak forms | Templates validated to sub-percent error on smooth cases and 2-5% on hard nonlinear ones; parser 60% valid on the first pass, 100% after retries, 97.1% field accuracy; custom geometry 90% |
| **PDE-Agents** (Adhikari, Noorsumar and Jensen, 2026) | FE solvers for PDEs | Simulation, analytics and database agents under a LangGraph supervisor, local open-source LLMs, a knowledge graph of materials, failure patterns and run history | Second-order convergence verified for the heat solver; 100% success on a 50-task ablation with the best retrieval mode; 97.8% success (57.6% first try) over 1,369 runs |
| **VFEAgent** (Peking University, 2026) | FEA from images and text | Vision-language agents extract a structured specification from drawings and descriptions (ReAct-style), then verification-first code synthesis with self-debugging and fallbacks | Higher rates of complete, physically valid simulations than LLM baselines on engineering mechanics scenarios |

**The spectrum.** From the LLM writing all the code (MechAgents) to the LLM writing input decks (FeaGPT, MooseAgent, CAX-Agent) to the LLM only filling a typed specification for validated templates (Penn State). Reliability increases along that spectrum; flexibility decreases. Production systems for engineering decisions sit at the constrained end, with free generation reserved for clearly flagged exploration.

## Lessons across systems

1. **Executable ≠ correct.** Report physics error against ground truth and silent wrong answers, not only completion. See [How to evaluate a simulation agent](llm-agents-for-simulation.md#how-to-evaluate-a-simulation-agent).
2. **Structure beats prompting:** typed specifications, dependency-aware generation, stage-specific retrieval.
3. **Recovery loops are the biggest lever** for completion rates (CAX-Agent: 69% → 93%), and the orchestrator, not the model, should own retry budgets.
4. **Open-source solvers are easier targets** (more public examples); proprietary APIs are low-resource languages for LLMs.
5. **Benchmarks are small and simple:** 8 to 110 tasks, mostly tutorial-like geometry, many 2025-2026 preprints. Treat percentages as indicative.
6. **Verification is the open problem:** independent checks (equilibrium, convergence, analytical bounds, energy balance) are rare in published systems and are where a new system can stand out.

## Commercial agentic features

- **Ansys 2026 R1** (March 2026) introduced the portfolio's first agentic capabilities, including a Mesh Agent in Mechanical (helping debug meshing failures) and a Discovery Validation Agent in early evaluation (flagging set-up issues), alongside the Engineering Copilot assistant.
- **Siemens** extended Simcenter's AI features (PhysicsAI) and announced AI copilots across its software portfolio.
- **Start-ups** such as Navier pitch "agent-driven engineering" between CAD and simulation.

See [Industry](../industry.md) for the companies and products.

## In an interview

**Thirty-second answer to "What did FEABench measure, and what did it find?"** "It tests whether LLMs and agents can solve finite-element problems end to end by operating COMSOL through its API, on 15 hand-verified problems with a computable target value. An agent with execution feedback and documentation tools raised executable code from 62% to 88%, but only one problem came within 10% of the answer, and that was COMSOL's default temperature, so effectively none was solved correctly. The lesson is that executability is easy to improve and correctness is the real problem, which is why I would constrain the LLM to specifications and verify every result."

**Thirty-second answer to "How do Foam-Agent, MetaOpenFOAM and OpenFOAMGPT automate CFD?"** "All three generate OpenFOAM case files from a prompt with retrieval over tutorials and run a repair loop on errors. MetaOpenFOAM uses a MetaGPT-style assembly line of roles; OpenFOAMGPT is retrieval plus iterative correction with GPT-4o and o1 and stresses human oversight; Foam-Agent adds stage-specific multi-index retrieval, dependency-ordered file generation for cross-file consistency, and a reviewer conditioned on its past attempts, reporting 88% success on 110 tasks against 56% for MetaOpenFOAM. None of them proves physical accuracy beyond the test cases."

**Likely follow-ups**

- *How do agents automate FEA set-up?* From free code generation to input decks to constrained specifications for validated templates; recovery ladders like CAX-Agent's do the heavy lifting.
- *What would you add to these systems?* Independent verification gates, analysis-type selection with justification, and an evaluation on realistic geometries with silent-error counts.

## References

- [FEABench: Evaluating Language Models on Multiphysics Reasoning Ability](https://arxiv.org/abs/2504.06260) and [code](https://github.com/google/feabench)
- [MetaOpenFOAM](https://arxiv.org/abs/2407.21320), [OpenFOAMGPT](https://arxiv.org/abs/2501.06327), [Foam-Agent](https://arxiv.org/abs/2505.04997) and [Foam-Agent 2.0](https://arxiv.org/abs/2509.18178)
- [MechAgents](https://arxiv.org/abs/2311.08166), [FeaGPT](https://arxiv.org/abs/2510.21993), [MooseAgent](https://arxiv.org/abs/2504.08621)
- [CAX-Agent](https://arxiv.org/abs/2605.15218), [Constrained natural-language interface for FEniCS](https://arxiv.org/abs/2606.10928)
- [PDE-Agents](https://arxiv.org/abs/2606.07850), [VFEAgent](https://arxiv.org/abs/2605.28978)
- [Synopsys: Ansys 2026 R1 release](https://news.synopsys.com/2026-03-11-Synopsys-Launches-Ansys-2026-R1-to-Re-Engineer-Engineering-with-Joint-Solutions-and-AI-Powered-Products)
