# Text-to-CAD

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Text-to-CAD systems turn a natural-language description ("a 100 × 60 × 8 mm plate with four 8 mm holes 12 mm from the corners") into an **editable, parametric CAD model**, not just a mesh. Two families exist. **Sequence models** treat a CAD model as a sequence of construction commands (sketch lines and arcs, then extrude) and generate it with a transformer: DeepCAD (2021) introduced the representation and a dataset of 178,238 models; **Text2CAD** (NeurIPS 2024) added about 660,000 text annotations at levels from beginner to expert for about 170,000 of them. **Code-generation models** have an LLM write a CAD script in a real language, usually **CadQuery** (Python on the Open CASCADE kernel): **Text-to-CadQuery** (2025) fine-tuned open LLMs on 170,000 CadQuery annotations and raised exact-match accuracy from 58.8% to 69.3%; **CAD-Recode** (ICCV 2025) turns point clouds into CadQuery code with a small LLM trained on a million procedurally generated programs. Because generated code can run but be wrong, the strongest systems **verify**: **CADCodeVerify** (ICLR 2025) renders the result, has a vision-language model ask and answer questions about it, and feeds the answers back to fix the code. Today these work for simple prismatic parts (sketch-and-extrude brackets, plates, flanges) and are unreliable for complex, constraint-rich or free-form geometry; for simulation agents, **parametric templates** with checked dimensions remain more dependable than free-form generation.

**On this page**

- [Why CAD, not meshes](#why-cad-not-meshes)
- [Command-sequence models: DeepCAD and Text2CAD](#command-sequence-models-deepcad-and-text2cad)
- [CAD as code: CadQuery generation](#cad-as-code-cadquery-generation)
- [Verifying generated geometry](#verifying-generated-geometry)
- [Limits and how text-to-CAD fits a simulation workflow](#limits-and-how-text-to-cad-fits-a-simulation-workflow)
- [Code: execute a CAD program and check it against the request](#code-execute-a-cad-program-and-check-it-against-the-request)
- [In an interview](#in-an-interview)

## Why CAD, not meshes

A mesh or point cloud is a frozen shape; a parametric CAD model is a **recipe**: sketches with dimensions and constraints, features (extrude, revolve, fillet, hole, pattern), and a history. Engineers edit dimensions, manufacturing drawings come from it, and simulation pre-processors need its clean faces (to apply loads and boundary conditions to named surfaces). Generating CAD rather than a mesh means the output is directly usable and editable. See [CAD models and CAD datasets](geometry-representations.md#cad-models-and-cad-datasets).

## Command-sequence models: DeepCAD and Text2CAD

**DeepCAD (Wu, Xiao and Zheng, ICCV 2021).**

- Represents a CAD model as a sequence of operations, drawing an analogy with natural language: sketch commands (line, arc, circle, with quantized parameters) and extrusions (with direction, extent and boolean type).
- A transformer autoencoder learns to encode and generate such sequences; the paper demonstrated shape autoencoding and random shape generation.
- Released a dataset of **178,238** models with their construction sequences, collected from Onshape, now the standard training set for this family.

**Text2CAD (Khan et al., NeurIPS 2024 spotlight).**

- The first framework for **text-conditioned parametric CAD generation** at multiple levels of description.
- Built an annotation pipeline with vision-language and language models (Mistral and LLaVA-NeXT) that described DeepCAD models at four levels, from abstract ("a ring") to expert (exact dimensions and operations): about **660,000** annotations for about **170,000** models.
- An autoregressive transformer generates the command sequence from the text.

**Strengths:** compact, always-valid command grammar; quantized parameters make generation tractable. **Weaknesses:** limited to the operations in the grammar (mostly sketch and extrude in DeepCAD), quantization limits dimensional precision, and the output needs conversion to a real CAD kernel.

## CAD as code: CadQuery generation

Instead of a custom command format, let an LLM write code in an existing parametric CAD language, which it can run, debug and edit:

```py
import cadquery as cq
plate = (cq.Workplane("XY").box(100, 60, 8)
         .faces(">Z").workplane()
         .rect(76, 36, forConstruction=True).vertices().hole(8))   # four 8 mm holes, 12 mm from the edges
```

**Why code works well with LLMs:** they have seen a lot of Python; the code is human-readable and editable; it runs on a real kernel (Open CASCADE) producing STEP files; and execution errors give feedback.

**Systems.**

- **Text-to-CadQuery (Xie and Ju, 2025):** augmented the Text2CAD data with **170,000 CadQuery** annotations and fine-tuned six open LLMs to generate CadQuery directly from text, skipping the intermediate command representation. The best model improved top-1 exact match from 58.8% to 69.3% and reduced Chamfer distance (a geometric error between generated and target shapes) by 48.6%.
- **CAD-Recode (Rukhovich et al., ICCV 2025):** reverse engineering, from a **point cloud** to CadQuery code, using a relatively small LLM (Qwen2-1.5B) as the decoder with a lightweight point-cloud projector, trained on **one million procedurally generated** CAD programs. It outperformed earlier methods on DeepCAD, Fusion 360 and the real-world CC3D datasets, and its output can be edited or queried by off-the-shelf LLMs.
- **Reinforcement learning with geometric rewards** (CAD-Coder and others, 2025) fine-tunes generators with a reward based on Chamfer distance to the target, after supervised training.
- **Commercial tools:** Zoo offers text-to-CAD through an API on its own geometry engine and CAD language; CAD vendors are adding generative assistants (Ansys GeomAI for concept geometry, for example).

## Verifying generated geometry

Generated code can execute and still produce the wrong part: a radius used as a diameter, a hole on the wrong face, a missing fillet, units confused. Verification closes the loop.

**CADCodeVerify (Alrashedy et al., ICLR 2025).**

1. An LLM generates CadQuery code from a prompt.
2. The code is executed and the object is **rendered** from several views.
3. A vision-language model **generates validation questions** from the prompt ("Does the plate have four holes?", "Are the holes near the corners?") and answers them by looking at the renders.
4. The answers become feedback for the next code revision.

They also introduced **CADPrompt**, a benchmark of 200 prompts with expert-written CadQuery solutions. With GPT-4, the verification loop reduced point-cloud distance by 7.30% and improved the success rate by 5.0% over prior approaches.

**Programmatic checks** complement visual ones and are stricter: measure the executed solid (bounding box, volume, number and size of holes, wall thicknesses, watertightness and validity of the B-rep) and compare with the request, as the [code below](#code-execute-a-cad-program-and-check-it-against-the-request) does. Manufacturing rules (minimum wall thickness, hole-to-edge distance, draft angles) can be checked the same way.

## Limits and how text-to-CAD fits a simulation workflow

**Current limits.**

- Good on **simple prismatic parts**; weak on complex assemblies, free-form surfaces, tight constraints, and design intent (which dimensions are driving).
- **Dimensional precision** and consistency with the text are not guaranteed; benchmarks report shape similarity, which can hide a wrong critical dimension.
- Datasets are dominated by sketch-and-extrude models from DeepCAD; real industrial parts have fillets, drafts, shells, patterns and tolerances.
- **Simulation readiness** is rarely considered: clean, named faces for loads and supports, no slivers, defeaturing.

**In a simulation agent:**

1. Use **parametric templates** (a library of CadQuery functions for brackets, plates, cylinders, layer stacks) for the common cases; the LLM only fills parameters.
2. Allow generated geometry only as a fallback for parts outside the catalogue, executed in a sandbox and verified (bounding box, volume, features, validity) before meshing.
3. **Name faces** in the template or generated code, so the agent can apply boundary conditions to "fixed_face" and "load_face" instead of guessing.
4. Show the rendered geometry and the measured dimensions to the engineer before running expensive simulations.

The Penn State FEniCS system follows exactly this pattern for meshing: a catalogue of geometries plus generated gmsh code only for non-catalogue shapes, reaching 90% success on custom geometry. See [FEA agents](simulation-agents-research.md#fea-agents).

## Code: execute a CAD program and check it against the request

A minimal stand-in for the verification step. A CAD "program" (a box, then holes through it) is executed into a voxel solid, its properties are measured, and they are compared with the request; the first draft contains a typical slip (the hole radius written where the diameter belongs).

```python
import math
import numpy as np

def execute(program):
    """Execute a tiny sketch-and-extrude program into a voxel solid and measure it (units: mm).
    Operations: ("box", sx, sy, sz) from the origin; ("hole", cx, cy, diameter) through the full thickness."""
    _, sx, sy, sz = program[0]
    res = 0.25                                                     # voxel size
    x, y = np.meshgrid(np.arange(res / 2, sx, res), np.arange(res / 2, sy, res), indexing="ij")
    solid = np.ones_like(x, dtype=bool)
    holes = [op for op in program[1:] if op[0] == "hole"]
    for _, cx, cy, dia in holes:
        solid &= (x - cx) ** 2 + (y - cy) ** 2 > (dia / 2) ** 2
    return {"volume_mm3": solid.sum() * res * res * sz, "bbox_mm": (sx, sy, sz), "n_holes": len(holes),
            "min_hole_dia_mm": min(op[3] for op in holes),
            "min_edge_margin_mm": min(min(cx, sx - cx, cy, sy - cy) - d / 2 for _, cx, cy, d in holes)}

def check_against_request(props, request):
    """Compare the measured geometry with what was asked for, plus one manufacturing rule."""
    L, W, T, n, d = request["L"], request["W"], request["T"], request["n_holes"], request["hole_d"]
    expected_volume = L * W * T - n * math.pi * (d / 2) ** 2 * T
    return {"bounding box": props["bbox_mm"] == (L, W, T),
            "hole count": props["n_holes"] == n,
            "hole diameter": abs(props["min_hole_dia_mm"] - d) < 1e-9,
            "volume within 1%": abs(props["volume_mm3"] - expected_volume) <= 0.01 * expected_volume,
            "hole-to-edge margin >= 1 diameter": props["min_edge_margin_mm"] >= d}

# "A 100 x 60 x 8 mm plate with four 8 mm holes, 12 mm from each edge at the corners"
request = {"L": 100, "W": 60, "T": 8, "n_holes": 4, "hole_d": 8}
corners = [(12, 12), (88, 12), (12, 48), (88, 48)]
draft = [("box", 100, 60, 8)] + [("hole", cx, cy, 4) for cx, cy in corners]    # slip: the radius written as the diameter
fixed = [("box", 100, 60, 8)] + [("hole", cx, cy, 8) for cx, cy in corners]
for name, program in [("first draft", draft), ("after feedback", fixed)]:
    props = execute(program)
    failed = [k for k, ok in check_against_request(props, request).items() if not ok]
    print(f"{name:>14}: volume {props['volume_mm3']:.0f} mm^3, failed checks: {failed or 'none'}")
```

Output:

```
   first draft: volume 47584 mm^3, failed checks: ['hole diameter', 'volume within 1%']
after feedback: volume 46376 mm^3, failed checks: none
```

The draft runs and looks almost right (four holes, correct plate), but the measured hole diameter and volume give it away; the corrected program matches the expected volume (46,391.5 mm³; the voxel estimate is within 0.05%) and passes the edge-margin rule. In a real pipeline the execution step is CadQuery on Open CASCADE, the measurements come from the B-rep, and the failed checks are fed back to the LLM as text.

## In an interview

**Thirty-second answer to "How do text-to-CAD systems work?"** "Two approaches. Sequence models like DeepCAD and Text2CAD represent a part as a sequence of sketch and extrude commands and generate it with a transformer conditioned on text; Text2CAD added about 660,000 descriptions for 170,000 DeepCAD models. Code models have an LLM write CadQuery, Python on a real CAD kernel; Text-to-CadQuery fine-tuned open models on 170,000 CadQuery scripts and reached 69% exact match. Since code can run and still be wrong, systems like CADCodeVerify render the part and use a vision-language model, or programmatic measurements, to check it against the prompt and fix the code. They work for simple prismatic parts; for simulation I would use parametric templates and generate code only as a verified fallback."

**Likely follow-ups**

- *Why generate code rather than a mesh?* The result is editable, parametric, exact, and has faces you can name for boundary conditions.
- *How would you verify generated CAD?* Execute it, check validity and watertightness, measure bounding box, volume and features against the request, check manufacturing rules, and show renders to the engineer.
- *What is CAD-Recode?* Point cloud to CadQuery code with a small LLM and a point-cloud projector, trained on a million synthetic programs: reverse engineering into an editable model.

## References

- [DeepCAD: A Deep Generative Network for Computer-Aided Design Models (Wu, Xiao & Zheng, ICCV 2021)](https://arxiv.org/abs/2105.09492)
- [Text2CAD: Generating Sequential CAD Models from Beginner-to-Expert Level Text Prompts (Khan et al., NeurIPS 2024)](https://arxiv.org/abs/2409.17106)
- [Text-to-CadQuery (Xie & Ju)](https://arxiv.org/abs/2505.06507)
- [CAD-Recode: Reverse Engineering CAD Code from Point Clouds (Rukhovich et al.)](https://arxiv.org/abs/2412.14042)
- [Generating CAD Code with Vision-Language Models for 3D Designs (CADCodeVerify; Alrashedy et al., ICLR 2025)](https://arxiv.org/abs/2410.05340)
- [CadQuery](https://github.com/CadQuery/cadquery) and [Zoo API](https://zoo.dev/machine-learning-api)
