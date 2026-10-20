# LLM Agents for Simulation

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** An LLM agent for simulation turns a plain-language engineering request into a simulation set-up (geometry, mesh, materials, boundary conditions, solver settings), runs it through tools, reads the results and repairs failures. Research systems already do this for COMSOL, OpenFOAM, MOOSE, Ansys MAPDL, CalculiX and FEniCS, and vendors shipped their first agentic features in 2026. **What works today:** producing input files and API calls that *run*, for standard problems, with retrieval over documentation and a repair loop driven by solver error messages; Foam-Agent reports 88% success on 110 OpenFOAM tasks, CAX-Agent 93% completion on 50 MAPDL tasks. **Where they fail:** results that run but are *wrong* (boundary conditions, units, signs, physics choices), unusual geometry, and knowing when the model is the right idealization; on FEABench's verified COMSOL problems, the best agent produced executable code 88% of the time but solved no problem completely and correctly. Robust designs therefore keep the LLM **out of the numerics**: it fills a validated, structured specification; verified tools and templates do the physics; independent checks (equilibrium, mesh convergence, analytical bounds) stand between the solver and any reported number. The **ReAct** pattern (reason, act with a tool, observe the result, repeat) is the natural loop. Evaluate agents on a task suite with ground truth, and count **silent wrong answers**, runs that finish with a wrong number, as the headline metric.

**On this page**

- [What agents can do today and where they fail](#what-agents-can-do-today-and-where-they-fail)
- [Why the LLM should not write solver physics](#why-the-llm-should-not-write-solver-physics)
- [The ReAct pattern](#the-react-pattern)
- [A reference architecture](#a-reference-architecture)
- [How to evaluate a simulation agent](#how-to-evaluate-a-simulation-agent)
- [Code: a ReAct-style agent with a verification gate](#code-a-react-style-agent-with-a-verification-gate)
- [In an interview](#in-an-interview)

## What agents can do today and where they fail

**Reliable enough to be useful (with a harness):**

1. **Parsing a request into a structured specification:** geometry parameters, materials, loads, supports, analysis type, outputs. A constrained FEniCS interface (Upadhyay and Reinhart, Penn State, 2026) reached 100% valid parses after retries (60% on the first pass) and 97.1% field-level accuracy on a 15-prompt benchmark.
2. **Generating solver input for standard cases** using retrieval over tutorials and documentation: OpenFOAM dictionaries (Foam-Agent, MetaOpenFOAM, OpenFOAMGPT), MOOSE input files (MooseAgent, 93% average success), APDL scripts (CAX-Agent), CalculiX decks (FeaGPT).
3. **Repairing failures from error logs:** a reviewer or recovery loop reads the solver's error message and edits the input. In CAX-Agent's study, completion rose from 69.3% with no recovery to 92.7% with model-driven recovery.
4. **Simple custom geometry** through code (gmsh or CadQuery scripts): the Penn State system succeeded on 9 of 10 custom-geometry cases.
5. **Post-processing and reporting:** extracting results, plotting, writing summaries.
6. **Documentation assistance** inside commercial tools (Ansys Engineering Copilot, Siemens copilots).

**Where they fail:**

1. **Correctness, not executability.** Code that runs can apply a load to the wrong face, use mm with Pa, pick plane stress instead of plane strain, or leave a rigid-body mode. FEABench (Google, 2025) is the clearest evidence: executable COMSOL API calls 88% of the time, but only one of 15 verified problems produced a value within 10% of the answer, and that one was an accident (it exported COMSOL's default 20 °C against a correct 18.3 °C). Foam-Agent's authors also note that execution success does not guarantee physical fidelity.
2. **Physics decisions:** whether nonlinearity, contact, transient effects or turbulence modelling matter; the right idealization (symmetry, shell vs solid, which details to defeature).
3. **Meshing difficult geometry:** thin layers, contact regions, boundary layers; singularities that make peak stresses meaningless.
4. **Low-resource APIs:** proprietary scripting languages with little public code are harder than open-source tools with many examples.
5. **Long, open-ended workflows** without checkpoints: errors compound across stages.
6. **Benchmarks are still small and simple** (15-110 tasks, simple geometries), and many results are 2025-2026 preprints; headline success rates do not transfer to industrial models.

Details of the individual systems are on [Simulation agents research](simulation-agents-research.md).

## Why the LLM should not write solver physics

1. **Silent wrong answers are worse than crashes.** A crash is noticed; a plausible number from a wrong weak form, sign or boundary condition may be acted upon. Physics errors rarely produce error messages.
2. **Verification is hard after the fact.** Checking LLM-written solver code requires the expertise the agent was meant to replace, for every new run.
3. **Liability and traceability.** Engineering decisions must trace to verified methods; a report saying "the model wrote a new weak form" will not pass a quality review.
4. **Evidence.** In MechAgents (Ni and Buehler), agent teams that wrote FEniCS code made mistakes in weak forms and traction formulas that needed other agents to correct; the Penn State constrained interface instead confined the LLM to parsing and custom gmsh geometry and dispatched to five human-written, validated templates (linear elasticity, hyperelasticity, elastoplasticity, thermo-mechanics, phase-field fracture), which matched analytical and benchmark solutions to under 1% on smooth problems and 2-5% on hard nonlinear ones.

**The division of labour:**

| The LLM does | Verified tools do |
| --- | --- |
| Understand the request and ask clarifying questions | Geometry from parametric templates (or sandboxed, checked scripts) |
| Fill a typed specification with units | Unit normalization and schema validation |
| Choose among templates and justify the analysis type | Meshing with quality checks and convergence studies |
| Read error messages and propose fixes within a budget | Solving with validated templates |
| Explain results and caveats | Verification: equilibrium, convergence, analytical bounds, energy balance |

## The ReAct pattern

**ReAct (Yao et al., ICLR 2023)** interleaves **reasoning traces** and **actions**: the model writes a thought ("the load is given in kN; I need the cross-section"), takes an action (calls a tool, such as `validate_spec` or `run_solver`), receives an **observation** (the tool's output or error), and repeats until it can answer. The paper showed that combining reasoning with actions reduced hallucination compared with reasoning alone, because facts come from tools, and made the trajectory interpretable.

**Why it suits engineering workflows.**

- Every number in the answer comes from a tool call, not from the model's memory.
- The trace (thought, action, observation) is an **audit log** an engineer can review.
- Failures are observations: a solver error or a failed check becomes input to the next step.
- The orchestrator (plain code, not the LLM) can own the **step budget, retries and stop rules**, and force verification before any final answer.

**Variants:** plan-and-execute (plan all steps first, then execute), multi-agent role splits (planner, mesher, runner, reviewer, as in Foam-Agent), and tool protocols such as the Model Context Protocol (MCP), which Foam-Agent uses to expose its functions to other agents.

## A reference architecture

```
request (text or voice) ─► LLM: parse to a typed spec, ask if ambiguous
        ─► validator: schema, units, plausible ranges, constraints suppress rigid-body motion
        ─► geometry template ─► mesher (named faces, quality checks)
        ─► solver template (validated) ─► verification gate (equilibrium, convergence, analytical bounds)
        ─┬─► pass: report with assumptions, evidence table and limitations
         └─► fail: LLM proposes a fix from the error (bounded retries) ─► else escalate to an engineer
```

Design rules: the orchestrator owns retries; every stage reads and writes the same spec so any run can be replayed; generated code runs in a sandbox with a timeout; defaults the LLM fills in are written into the report. The system-design version of this is [Design: an LLM agent for structural checks](design-llm-structural-check-agent.md).

## How to evaluate a simulation agent

1. **A task suite with ground truth:** analytical solutions (cantilever deflection, Kirsch plate, Lamé cylinder, 1D conduction) and reference simulations, covering each template and physics type, with **paraphrases** of each request, **underspecified or contradictory** requests (the agent must ask, not guess), and unit traps.
2. **Metrics per stage:**
   - parsing: first-pass and final valid-spec rate, field-level accuracy, unit errors;
   - clarification: share of underspecified prompts where the agent asks;
   - execution: completion rate, retries per task, human interventions;
   - physics: relative error against ground truth at a converged mesh;
   - **silent wrong-answer rate**: runs that finish and report a number outside tolerance (the most important metric, target zero);
   - cost: wall-clock, tokens, GPU or CPU time.
3. **Repeat runs.** LLM outputs vary even at temperature zero; run each task several times and report means and spreads (CAX-Agent ran 50 tasks three times each, 450 case-runs across strategies).
4. **Ablations:** with and without retrieval, recovery loop, verification gate; different models.
5. **Blind human grading** where automatic checks are not possible: CAX-Agent used two blind raters with a quadratic weighted kappa of 0.84.
6. **Out-of-distribution tasks:** geometries and physics outside the templates must be refused or escalated, not answered.

## Code: a ReAct-style agent with a verification gate

A minimal, runnable version of the architecture above. The "LLM" is a scripted stand-in that parses requests and deliberately makes a common mistake (reading millimetres as metres); the tools are real: a schema and plausibility validator, a beam finite-element solver, and an independent verification gate (equilibrium, mesh convergence, the analytical solution PL³/3EI). The evaluation runs 40 tasks with known answers, with and without the gate.

```python
import re, json
import numpy as np

UNITS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "N": 1.0, "kN": 1e3, "GPa": 1e9, "MPa": 1e6}
SCHEMA = {"length_m": (0.01, 20), "width_m": (1e-3, 1.0), "height_m": (1e-3, 1.0), "load_N": (1e-3, 1e7), "E_Pa": (1e8, 1e12)}

def validate_spec(spec):
    """Schema and plausibility check of a cantilever spec in SI units. Returns a list of problems."""
    problems = [f"missing {k}" for k in SCHEMA if k not in spec]
    for k, (lo, hi) in SCHEMA.items():
        if k in spec and not lo <= spec[k] <= hi:
            problems.append(f"{k}={spec[k]:g} outside plausible range [{lo:g}, {hi:g}]: check units")
    return problems

def solve_cantilever_fe(spec, n_el=8):
    """Euler-Bernoulli beam FE (Hermite cubic elements), clamped at x = 0, tip load. Tip deflection and reaction."""
    L, P, E = spec["length_m"], spec["load_N"], spec["E_Pa"]
    I = spec["width_m"] * spec["height_m"] ** 3 / 12
    h = L / n_el
    k = E * I / h ** 3 * np.array([[12, 6*h, -12, 6*h], [6*h, 4*h*h, -6*h, 2*h*h],
                                   [-12, -6*h, 12, -6*h], [6*h, 2*h*h, -6*h, 4*h*h]])
    K = np.zeros((2 * n_el + 2, 2 * n_el + 2))
    for e in range(n_el):
        K[2*e:2*e+4, 2*e:2*e+4] += k
    f = np.zeros(2 * n_el + 2); f[-2] = -P
    u = np.zeros_like(f)
    u[2:] = np.linalg.solve(K[2:, 2:], f[2:])                              # clamp: w = w' = 0 at x = 0
    return {"tip_deflection_m": -u[-2], "reaction_N": (K @ u - f)[0]}

def verify(spec, result):
    """Independent checks before any number is reported."""
    I = spec["width_m"] * spec["height_m"] ** 3 / 12
    analytic = spec["load_N"] * spec["length_m"] ** 3 / (3 * spec["E_Pa"] * I)
    fine = solve_cantilever_fe(spec, n_el=16)["tip_deflection_m"]
    return {"equilibrium": bool(abs(result["reaction_N"] - spec["load_N"]) <= 1e-6 * spec["load_N"]),
            "mesh convergence": bool(abs(fine - result["tip_deflection_m"]) <= 1e-3 * abs(fine)),
            "analytical solution": bool(abs(analytic - result["tip_deflection_m"]) <= 1e-2 * analytic)}

def llm_parse(prompt, feedback=None, careless=False):
    """Stand-in for the LLM: text -> spec. When careless, it reads mm as m unless given feedback."""
    spec = {}
    for key, pattern in [("length_m", r"(\d+\.?\d*)\s*(mm|cm|m)\s+long"), ("width_m", r"(\d+\.?\d*)\s*(mm|cm|m)\s+wide"),
                         ("height_m", r"(\d+\.?\d*)\s*(mm|cm|m)\s+(?:deep|high|tall)"), ("load_N", r"(\d+\.?\d*)\s*(kN|N)\b"),
                         ("E_Pa", r"(\d+\.?\d*)\s*(GPa|MPa)")]:
        m = re.search(pattern, prompt)
        if m:
            value, unit = float(m.group(1)), m.group(2)
            spec[key] = value * (1.0 if (careless and unit == "mm" and not feedback) else UNITS[unit])
    return spec

def run_agent(prompt, use_gate=True, careless=False, max_steps=4, log=None):
    """ReAct-style loop owned by plain code: act (parse, validate, solve, verify), observe, retry within a budget."""
    feedback = None
    for step in range(max_steps):
        spec = llm_parse(prompt, feedback, careless)
        problems = validate_spec(spec) if use_gate else []
        if log is not None:
            log.append(f"step {step}: spec {json.dumps({k: round(v, 6) for k, v in spec.items()})} problems {problems}")
        if problems:                                                       # the observation becomes feedback
            feedback = problems
            continue
        result = solve_cantilever_fe(spec)
        if use_gate:
            checks = verify(spec, result)
            if log is not None:
                log.append(f"step {step}: checks {checks}")
            if not all(checks.values()):
                return {"status": "escalate to engineer", "checks": checks}
        return {"status": "answer", "tip_deflection_mm": float(f"{1e3 * result['tip_deflection_m']:.4g}")}   # 4 significant digits
    return {"status": "gave up", "problems": feedback}

trace = []
print(run_agent("Steel cantilever 2 m long, 50 mm wide, 100 mm deep, 10 kN at the tip, E = 210 GPa", careless=True, log=trace))
print("\n".join(trace))

# Evaluation: 40 tasks with known answers; count correct answers and silent wrong answers.
rng = np.random.default_rng(0)
tasks = []
for i in range(40):
    L, b, h, P = rng.uniform(0.5, 3), rng.uniform(20, 80), rng.uniform(40, 200), rng.uniform(1, 20)
    in_mm = i % 2 == 0                                                     # half the prompts give sizes in mm
    sizes = f"{b:.0f} mm wide, {h:.0f} mm deep" if in_mm else f"{b / 1000:.3f} m wide, {h / 1000:.3f} m deep"
    prompt = f"Cantilever {L:.2f} m long, {sizes}, {P:.1f} kN at the tip, E = 200 GPa"
    width = round(b) / 1000 if in_mm else round(b / 1000, 3)
    height = round(h) / 1000 if in_mm else round(h / 1000, 3)
    truth_mm = 1e3 * round(P, 1) * 1e3 * round(L, 2) ** 3 / (3 * 200e9 * width * height ** 3 / 12)
    tasks.append((prompt, truth_mm))
for gate in [False, True]:
    outcomes = {"correct": 0, "silent wrong answer": 0, "escalated or gave up": 0}
    for prompt, truth in tasks:
        out = run_agent(prompt, use_gate=gate, careless=True)
        if out["status"] != "answer":
            outcomes["escalated or gave up"] += 1
        elif abs(out["tip_deflection_mm"] - truth) <= 0.01 * truth:
            outcomes["correct"] += 1
        else:
            outcomes["silent wrong answer"] += 1
    print(f"verification gate {'on ' if gate else 'off'}: {outcomes}")
```

Output:

```
{'status': 'answer', 'tip_deflection_mm': 30.48}
step 0: spec {"length_m": 2.0, "width_m": 50.0, "height_m": 100.0, "load_N": 10000.0, "E_Pa": 210000000000.0} problems ['width_m=50 outside plausible range [0.001, 1]: check units', 'height_m=100 outside plausible range [0.001, 1]: check units']
step 1: spec {"length_m": 2.0, "width_m": 0.05, "height_m": 0.1, "load_N": 10000.0, "E_Pa": 210000000000.0} problems []
step 1: checks {'equilibrium': True, 'mesh convergence': True, 'analytical solution': True}
verification gate off: {'correct': 20, 'silent wrong answer': 20, 'escalated or gave up': 0}
verification gate on : {'correct': 40, 'silent wrong answer': 0, 'escalated or gave up': 0}
```

The trace shows the loop working as intended: the first parse reads "50 mm" as 50 m, the validator's plausibility check turns that into an observation, the retry parses correctly, and the answer (30.48 mm, equal to PL³/3EI) is released only after equilibrium, mesh-convergence and analytical checks pass. Without the gate, the same agent completes every task and is wrong on half of them, silently; with it, all 40 are correct. (A first version of this harness reported results rounded to 0.001 mm and scored one tiny deflection, 0.0385 mm, as wrong: reporting precision is part of correctness too.) In a real system, the parser is an LLM and the templates are FE codes, but the structure, and the value of the gate, are the same.

## In an interview

**Thirty-second answer to "What can LLM agents do in a CAE workflow today, and where do they fail?"** "With a good harness they can turn a request into a structured set-up, generate input files for standard cases using retrieval over documentation, run them, and fix errors from the logs: Foam-Agent reports 88% success on 110 OpenFOAM tasks. Where they fail is correctness: FEABench found 88% executable COMSOL code but no problem solved completely and correctly, because wrong boundary conditions, units or physics choices still run. So I keep the LLM to parsing and orchestration, use validated templates for the physics, and put an independent verification gate before any reported number."

**Likely follow-ups**

- *What is the ReAct pattern?* Interleaved reasoning, tool actions and observations; facts come from tools, the trace is auditable, and failures become inputs to the next step.
- *How would you evaluate the agent?* A task suite with ground truth, paraphrases and underspecified prompts; parsing, completion, physics error and above all the silent wrong-answer rate, over repeated runs, with ablations.
- *What would you never let the agent do?* Write weak forms or solver code for a production answer, or fill in a missing boundary condition without saying so.

## References

- [FEABench (Mudur et al.)](https://arxiv.org/abs/2504.06260)
- [CAX-Agent: a harness for Ansys MAPDL (2026)](https://arxiv.org/abs/2605.15218)
- [A constrained natural-language interface for FEniCS multiphysics (Upadhyay & Reinhart)](https://arxiv.org/abs/2606.10928)
- [Foam-Agent](https://arxiv.org/abs/2505.04997) and [Foam-Agent 2.0](https://arxiv.org/abs/2509.18178)
- [MechAgents (Ni & Buehler)](https://arxiv.org/abs/2311.08166)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al.)](https://arxiv.org/abs/2210.03629)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [CadQuery](https://github.com/CadQuery/cadquery) and [FEniCSx tutorial (Dokken)](https://jsdokken.com/dolfinx-tutorial/)
