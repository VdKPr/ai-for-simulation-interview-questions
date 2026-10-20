# Design: LLM Structural Check Agent

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design an LLM agent that runs a structural check from a plain-language request.

**Short answer.**

- **The LLM translates and orchestrates; it never produces the physics.** It turns the request ("Will this bracket hold 2 kN at the hole, steel, safety factor 1.5?") into a **structured, validated specification**: geometry reference, material, named load and support regions, magnitudes with units, required checks.
- **Ambiguities go back to the user.** It asks instead of guessing, and the user confirms the specification before anything runs.
- **Deterministic tools do the work:** CAD lookup, meshing, the FE solver and post-processing.
- **A verification gate checks the results** before the agent may report them:
  - mesh convergence;
  - equilibrium (reactions match loads);
  - displacement and stress plausibility;
  - an analytical estimate when one exists.
- **The report is generated from tool outputs only, with every number traceable.** A wrong-but-confident answer is the main failure mode, so the evaluation counts **silent wrong answers** separately from failures that are flagged.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [From request to specification](#from-request-to-specification)
- [Tools](#tools)
- [The agent loop](#the-agent-loop)
- [Verification gate](#verification-gate)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **Users:** design engineers who are not FEA specialists, or analysts who want automation?
- **Scope:** linear static checks on single parts first; assemblies, contact and nonlinear analysis later (each multiplies the failure modes).
- **Inputs:** CAD from the PDM system, a material library, a standard for checks (allowables, safety factors).
- **Output:** pass/fail with margin, stress plot, report; does an engineer sign off?
- **Environment:** which solver, licences, on-premises (CAD and results are sensitive).

## From request to specification

The LLM's job is extraction into a schema, which can be validated:

```py
{
  "part": "PDM:BRK-1042 rev C",
  "material": "S355 steel",                     # must match the material library
  "load_cases": [{"name": "LC1",
                  "loads": [{"region": "hole_A", "type": "force", "vector_N": [0, 0, -2000]}],
                  "supports": [{"region": "base_face", "type": "fixed"}]}],
  "checks": [{"type": "von_mises", "allowable": "yield / 1.5"}],
  "assumptions": ["linear elastic", "small deformation"]
}
```

- **Validation:** units present and convertible; material exists; regions exist in the CAD (named selections or faces picked by the user); loads non-zero; supports sufficient to prevent rigid-body motion.
- **Ambiguity → question:** "at the hole" with two holes, a missing direction, "steel" with no grade. The agent asks; it does not choose silently.
- **Confirmation:** the user sees the specification (and a picture of loads and supports on the part) and confirms before solving.

## Tools

| Tool | Deterministic? | Notes |
| --- | --- | --- |
| `get_cad(part_id)` | Yes | From PDM; returns geometry plus named regions |
| `material_props(name)` | Yes | Validated library; never LLM-recalled values |
| `mesh(geometry, size, refine_regions)` | Yes | Returns quality metrics |
| `solve_static(model)` | Yes | FE solver; returns status, displacements, stresses, reactions |
| `refine_and_resolve(...)` | Yes | For convergence checks |
| `analytical_estimate(...)` | Yes | Beam or plate formulas when the geometry allows |
| `report(results)` | Template | Numbers inserted from results, not generated |

Tool interfaces are typed and validated; the LLM calls them but cannot pass free text into solver input files. See [Why the LLM should not write solver physics](llm-agents-for-simulation.md#why-the-llm-should-not-write-solver-physics).

## The agent loop

A **ReAct** loop (reason, act with a tool, observe, repeat) with a fixed plan skeleton ([The ReAct pattern](llm-agents-for-simulation.md#the-react-pattern)):

1. Parse the request into the specification; ask questions; get confirmation.
2. Fetch CAD and material; validate.
3. Mesh with local refinement at load and support regions and fillets; check quality.
4. Solve; on solver errors (singular stiffness: insufficient supports), diagnose with rules, not guesses, and ask the user if the fix changes the problem.
5. Run the verification gate (below).
6. Post-process: maximum von Mises, location, margin against the allowable; displacement.
7. Generate the report from results; flag assumptions and limitations.

Step limits and timeouts prevent loops; every tool call is logged.

## Verification gate

The agent cannot report a result that fails any of these checks:

- **Equilibrium:** sum of reaction forces equals applied loads (within tolerance).
- **Mesh convergence:** peak stress at the critical location changes by less than a few percent between two mesh levels; if not, refine, or report as not converged. Singular points (sharp corners, point loads) are identified and excluded from the check, with a note ([Singularities](stress-concentrations-and-singularities.md)).
- **Plausibility:** displacement small relative to part size (linear assumption); stresses within expected magnitude.
- **Analytical cross-check** where possible (a cantilever estimate for a simple bracket).
- **Unit check:** results consistent with input units.

The demo in [LLM agents for simulation](llm-agents-for-simulation.md#code-a-react-style-agent-with-a-verification-gate) shows the effect: without the gate, half of the answers on a test set were silently wrong; with it, the errors were caught and the answers corrected.

## Evaluation

- **Test suite** of requests with known answers: simple geometries with analytical solutions, and real parts with analyst-verified results; include ambiguous and impossible requests.
- **Metrics:**
  - correct answers;
  - flagged failures (the agent says it cannot or asks);
  - **silent wrong answers** (the critical metric; target zero);
  - clarification quality (did it ask when it should, and not when it need not?);
  - steps and cost per request.
- **Robustness:** paraphrased requests, unit variations (kN, N, lbf), irrelevant details.
- **Regression testing** whenever the LLM, prompts or tools change ([How to evaluate a simulation agent](llm-agents-for-simulation.md#how-to-evaluate-a-simulation-agent)).
- Public benchmarks for context: FEABench ([Simulation agents research](simulation-agents-research.md)).

## Architecture

```
request ─► LLM: extract spec ─► schema validation ─► questions? ─► user confirms spec (+ picture)
                                                                         │
            tools: CAD ─► material ─► mesh ─► solve ─► verification gate (equilibrium, convergence, plausibility)
                                                              │ fail → refine / diagnose / ask
                                                              ▼ pass
                                 report from results (template) ─► engineer sign-off ─► audit log
```

## Risks and roadmap

- **Silent wrong answers** from misinterpreted requests: confirmation step plus the gate.
- **Over-reliance:** users treat the report as certified. The report states assumptions, and sign-off stays with an engineer.
- **Prompt injection** through documents or CAD metadata: tools accept only typed, validated inputs.
- **v0:** single parts, linear static, one load case, analytical-solution test suite.
- **v1:** multiple load cases, PDM integration, convergence automation.
- **v2:** assemblies with bonded contact; surrogate pre-screening before FEA.

## In an interview

**Thirty-second answer.** "The LLM's job is translation and orchestration, not physics. It turns the request into a structured specification: part, material from a validated library, named load and support regions, magnitudes with units and the checks required. It asks when anything is ambiguous, and the user confirms the specification with a picture of loads and supports. Deterministic tools then mesh, solve and post-process. Before the agent may report, a verification gate checks equilibrium, mesh convergence at the hot spot and plausibility. The report is generated from tool outputs, so every number is traceable. I'd evaluate it on a test suite with known answers and track silent wrong answers as the key metric."

**Likely follow-ups**

- *What if the user gives a load in the wrong units?* Units are required in the schema; implausible magnitudes trigger a question.
- *Could the LLM pick mesh sizes?* It can propose them; the convergence check decides.

## References

- S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023 ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)).
- N. Mudur et al., "FEABench: Evaluating Language Models on Multiphysics Reasoning Ability", 2025 ([arXiv:2504.06260](https://arxiv.org/abs/2504.06260)).
- [LLM agents for simulation](llm-agents-for-simulation.md), [Simulation agents research](simulation-agents-research.md)
