# Behavioural Questions

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Behavioural rounds in AI-for-simulation companies check whether you can turn technical work into results other people rely on, whether you are honest when things go wrong, and whether you can communicate with both ML people and simulation engineers. Prepare **five or six real stories** and adapt them to the question. Tell each with **STAR** (Situation, Task, Action, Result), spend most of the time on *your* actions and decisions, put a number on the result where you can, and end with what you learned. For the presentation round, prepare one project as a structured 20–30 minute talk and expect questions on every number. For "why us", be specific about the company's products and papers; for "how do you keep up", name sources and something you actually reimplemented.

**On this page**

- [Preparing your stories](#preparing-your-stories)
- [Research that people actually used](#research-that-people-actually-used)
- [Finding out your own results were wrong](#finding-out-your-own-results-were-wrong)
- [Presenting a technical project](#presenting-a-technical-project)
- [Why this company, and why AI for simulation](#why-this-company-and-why-ai-for-simulation)
- [Keeping up with the field](#keeping-up-with-the-field)
- [In an interview](#in-an-interview)

## Preparing your stories

Write each story down in four or five lines, then practise saying it in about two minutes.

| Story | Questions it answers |
| --- | --- |
| A result of yours that turned out to be wrong | Mistakes, honesty, rigour, testing |
| Research or a prototype that people used | Impact, ownership, working with users |
| A disagreement settled with evidence | Conflict, working with domain experts |
| A trade-off: speed vs accuracy, proof of concept vs product | Judgement, prioritization |
| Explaining a limitation to a non-specialist | Communication, customer-facing work |
| A project you scoped yourself | Ownership, ambiguity, planning |

**STAR, done well.**

- **Situation and task** in two or three sentences: enough context for the stakes to be clear.
- **Action** is most of the answer: what *you* did ("I", not "we"), what alternatives you considered, and why you chose what you did.
- **Result** with a number where possible (time saved, error reduced, number of users), and honestly scaled: "the team adopted it" is better than an invented percentage.
- **Learning:** what you would do differently now.

**Confidentiality.** Many stories come from employers or clients. Describe the problem in generic terms ("a thermal model of an industrial component"), use rounded or textbook numbers, and never name confidential data, customers or results. Interviewers respect someone who protects a client's information; it is a preview of how you would treat theirs.

## Research that people actually used

**What they are testing.** Whether you can go beyond a notebook result: understand what users need, make it reliable, and get it adopted. Physics-AI companies live or die on this.

**A strong story includes:**

- **who the users were** and what problem they had (time, cost, a decision they could not make);
- **how you found out what they needed:** you talked to them, watched their workflow, or delivered something small early;
- **what you changed** to make it usable: an interface in their tools, validation they trusted, documentation, error handling, speed;
- **evidence of use:** how many people, how often, what decision it changed;
- **what you learned** about the gap between research and use.

**Template.**

> *Situation:* Engineers ran [an expensive simulation] for every design variant, which took [time].
> *Task:* I was asked to see whether a [surrogate / script / model] could cut that.
> *Action:* I built a first version in [weeks], tested it against [their held-out cases], and found that engineers would not use it until [it fitted their tool / gave uncertainty / handled their file format]. I added [that], wrote [docs or a validation report], and sat with two users while they tried it.
> *Result:* It was used for [screening of N designs per week], cutting [time] from [X] to [Y]. Final designs were still verified with the solver.
> *Learning:* Adoption depended more on integration and trust than on the last few percent of accuracy.

## Finding out your own results were wrong

**What they are testing.** Integrity and rigour. In this field, silent errors are the main risk, so the interviewer wants someone who finds their own mistakes, says so quickly, and changes the process so it cannot happen again.

**Common real causes worth knowing** (and good material for a story):

- **Data leakage:** near-duplicate designs or time steps in both training and test sets ([Data splits and leakage](data-splits-and-leakage.md)).
- **Normalization with test statistics**, or a units mismatch (millimetres vs metres).
- **Wrong metric:** a mean error that hid failures at the peak stress location ([Evaluation metrics](evaluation-metrics.md)).
- **An unconverged or mis-set-up simulation** in the training data.
- **A bug in post-processing:** inward normals flipping the sign of a force ([Mesh data processing](mesh-data-processing.md)).

**A strong story includes:** how you noticed (a sanity check, a physical argument, a result too good to be true), what you did immediately (told the people relying on it, withdrew the result), how you found the root cause, and what you changed permanently (a test, a check, a review step).

**Model answer outline.** "My model's test error looked much better than expected. That made me suspicious, so I checked the split and found that variants of the same base design were in both training and test sets. I told my lead the same day that the number was wrong, re-split by design family, and the honest error was about twice as high. I added a grouped split and a duplicate check to our pipeline. What I took from it: when a result looks too good, check the evaluation before celebrating."

## Presenting a technical project

Reported as a final-round format at PhysicsX ([Industry notes](../industry.md#reported-interview-formats)). Usually 20–30 minutes plus questions, to a mixed audience.

**Structure.**

1. **Problem and why it mattered** (one slide): the decision or cost at stake.
2. **Constraints:** data available, compute, accuracy needed, time.
3. **Approach, with alternatives considered** and why you rejected them.
4. **Results against baselines**, with the evaluation method stated (splits, metrics).
5. **What failed** and what you learned; where the method should not be used.
6. **Next steps** and what you would do with more time or data.

**Tips.**

- Choose a project where **you** made the key decisions and can answer deep questions.
- Show a baseline. A result without one invites "compared to what?".
- Prepare for questions on every number: how was it measured, on what split, with what variance?
- Keep confidential work generic (see above). If the work cannot be shown, present a public or personal project instead.
- Rehearse to time; leave at least a third of the slot for questions.

## Why this company, and why AI for simulation

**Why this company.** Be specific; generic praise sounds like every other candidate. Mention:

- their **product** and what it does (see [Industry notes](../industry.md));
- a **paper, model or release** you read, and what you found interesting or would ask about;
- their **customers or industries**, and how your background fits them;
- the **kind of role**: forward-deployed, research, platform. Say why that suits you.

**Why AI for simulation.** Connect your own path to the problem: simulation is accurate but slow, design needs many iterations, and ML can make exploration fast while solvers still guarantee the final answer. Mention what you bring from both sides: numerical methods and physics, and ML engineering.

**Model answer outline.** "I've worked on both sides, running FEA and CFD and building ML models, and the bottleneck I kept seeing was simulation time limiting how many designs engineers could look at. Your approach of [specific product or model] addresses exactly that, and I liked that your [paper or demo] is explicit about uncertainty and where the model shouldn't be trusted. The forward-deployed role suits me because I enjoy working directly with engineers on their problems."

## Keeping up with the field

Name concrete sources and show that you go beyond reading:

- **Papers:** arXiv listings (cs.LG and physics.comp-ph), proceedings of NeurIPS, ICML and ICLR (AI for science workshops), and *Journal of Computational Physics* and *CMAME* for the numerical side.
- **Benchmarks and code:** PDEBench, The Well, AirfRANS, DrivAerML; repositories such as PhysicsNeMo and neuraloperator ([Resources](../resources.md)).
- **Courses:** the ETH Zurich *AI in the Sciences and Engineering* course; Steve Brunton's lectures on data-driven science.
- **Company research:** blogs and papers from NVIDIA, Emmi AI, PhysicsX, Luminary and others.
- **Practice:** reimplement one paper's core idea at small scale (a Fourier layer, a message-passing layer, a PINN), which is the best way to understand what a paper actually claims.

**Model answer outline.** "I follow arXiv for physics-ML papers and a few research groups and company blogs, but what helps most is reimplementing ideas on a small problem. For example, I recently [reimplemented X on a 1D problem] and found that [an observation]. I also check claims against benchmarks, because results in this field depend heavily on the dataset and split."

## In an interview

**Thirty-second answer to "Tell me about yourself."** Give a two-minute version that links past to future: your background in simulation and ML, one achievement with a result, and why this role is the natural next step. End with a hook the interviewer can ask about.

**Likely follow-ups**

- *What would you do differently?* Always have an honest answer; "nothing" is a weak answer.
- *What was your specific contribution?* Separate your work from the team's.
- *What's a weakness?* Name a real one, with what you are doing about it.

## References

- [PhysicsX interview reports (Glassdoor)](https://www.glassdoor.com.mx/Entrevista/PhysicsX-Entrevista-E9306867-RVW97800229.htm) (presentation round, technical task)
- [Neural Concept forward-deployed engineer role](https://builtinbengaluru.in/job/application-engineer/8741962) (customer-facing communication)
- [Customer scenarios](customer-scenarios.md) and [Industry notes](../industry.md)
