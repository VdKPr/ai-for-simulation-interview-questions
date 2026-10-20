# Production and Deployment

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A deployed surrogate is a **pipeline**, not a model file: ingest and validate the customer's geometry, preprocess it exactly as in training, run inference on GPUs, post-process into quantities and fields the engineer's tools can open, attach uncertainty and an out-of-distribution flag, fall back to the solver when needed, and log everything. **Version data together with models** (a model is only reproducible if you know exactly which simulations, solver settings and preprocessing it saw). Guard every release with a **regression test suite**: golden cases with reference results, invariance tests (remeshing, point order, units), physics checks and latency budgets. Package models with **TorchScript**, **torch.export** or **ONNX** (and always check numerical parity after export), ship them in **Docker** images and run them on **Kubernetes** (inference services) or HPC schedulers (data generation). Track experiments and promote models through a **registry**. Integrate into the engineer's CAE workflow (plugins, VTK outputs, APIs) rather than asking them to change tools. **Monitor** input drift, OOD rates and spot-check errors to trigger retraining. When customers will not upload CAD, deploy **on-premises or in their cloud**, fine-tune on their site, or work from derived representations.

**On this page**

- [The deployment pipeline](#the-deployment-pipeline)
- [Versioning data together with models](#versioning-data-together-with-models)
- [A regression test suite for a surrogate](#a-regression-test-suite-for-a-surrogate)
- [ONNX vs TorchScript vs torch.export](#onnx-vs-torchscript-vs-torchexport)
- [Docker and Kubernetes](#docker-and-kubernetes)
- [Experiment tracking and model versions](#experiment-tracking-and-model-versions)
- [Integrating into an engineer's CAE workflow](#integrating-into-an-engineers-cae-workflow)
- [Monitoring and retraining](#monitoring-and-retraining)
- [When customers will not upload their CAD](#when-customers-will-not-upload-their-cad)
- [Code: export, parity checks and a regression suite](#code-export-parity-checks-and-a-regression-suite)
- [In an interview](#in-an-interview)

## The deployment pipeline

```
CAD / STL / mesh ─► validation (units, watertight, orientation, size) ─► preprocessing (sampling, SDF, features)
     ─► model inference (GPU service) ─► post-processing (forces, fields → VTP/VTU)
     ─► uncertainty + OOD gate ─┬─► results in the engineer's tool (ParaView, CAE plugin, report)
                                └─► solver fallback queue (HPC) ─► new verified data ─► retraining
```

Every arrow is code that must be tested and versioned: a preprocessing difference between training and serving (different point sampling, unit conversion, normalization statistics) is the most common cause of silently wrong production predictions ("training-serving skew").

## Versioning data together with models

A surrogate's behaviour depends on its training data as much as on its code. To reproduce or audit a prediction you need to know:

- **Which simulations:** case identifiers and file checksums, the experiment plan that produced them, which runs were excluded and why.
- **How they were produced:** solver and version, turbulence or material models, mesh settings, convergence criteria, post-processing scripts.
- **How they were preprocessed:** sampling, normalization statistics, features, splits, all with the code version.
- **The resulting dataset version:** an immutable identifier, for example a content hash of the manifest.

**Tools:** DVC or lakeFS (Git-like versioning for large files on object storage), Delta Lake or Iceberg tables for metadata, or simply immutable, content-addressed dataset directories plus a manifest in Git. The model's record (its "model card") then stores the dataset version, the code commit, the configuration, the metrics, and the **valid input ranges**.

**Why it matters in practice.** When a customer asks why the drag prediction for a design changed between releases, you can answer: new training data from solver version 2.1, which changed the wall-function treatment. When a bug is found in a preprocessing script, you can find every model affected. When a regulator or a customer's quality team audits, the chain from prediction back to simulations exists. See [Simulation file formats](simulation-file-formats.md#handling-terabytes-of-simulation-files).

## A regression test suite for a surrogate

Run on every candidate model in CI, before it can be promoted.

1. **Golden cases.** A fixed set of designs (including edge cases and previously reported failures) with reference solver results. Fail if any metric degrades beyond agreed thresholds, overall and per slice (per design family, per regime). Include the integrated quantities customers care about (drag, peak stress), not only field errors.
2. **Invariance and equivariance tests.** Point or node order, remeshing and resampling of the same geometry, unit changes (mm vs m inputs converted correctly), rigid translation, rotation and mirroring where the physics allows. The outputs must stay within tolerance.
3. **Physics and sanity checks.** Finite outputs; values in physical ranges; conservation or divergence-free where expected; force balance; monotonic trends the engineers know (more load gives more stress).
4. **Out-of-distribution behaviour.** Known out-of-range inputs must trigger the OOD flag and the fallback, not a confident answer.
5. **Uncertainty calibration** on a held-out set: coverage of stated intervals. See [Checking calibration](uncertainty-quantification.md#checking-calibration).
6. **Export parity.** The exported artefact (ONNX, TorchScript, TensorRT engine) gives the same outputs as the training model within tolerance, on several input sizes.
7. **Performance budgets.** Latency per request at typical and maximum mesh sizes, GPU memory, throughput.
8. **Compatibility.** The serving API's input and output schemas, file formats and units are unchanged (or versioned).

The [code below](#code-export-parity-checks-and-a-regression-suite) shows such a suite catching a real problem: the drag prediction changes by 16% when the same surface is sampled with half the points.

## ONNX vs TorchScript vs torch.export

| | TorchScript | torch.export | ONNX |
| --- | --- | --- | --- |
| What it produces | A serialized TorchScript program (traced or scripted) | An `ExportedProgram`: a full graph in PyTorch's ATen operator set, with symbolic shapes | A framework-neutral graph in the ONNX standard |
| Runs in | LibTorch (C++), Python without the model code | PyTorch, AOTInductor (compiled shared library), ExecuTorch for edge devices, and as the front end of the ONNX exporter | ONNX Runtime, TensorRT, OpenVINO and other runtimes |
| Status | In maintenance mode; PyTorch recommends torch.export for new work | The recommended export path in recent PyTorch | Mature, widely supported standard |
| Dynamic shapes | Tracing bakes in shapes and control flow; scripting handles control flow | Explicit dynamic dimensions (for example a variable number of points) | Dynamic axes |
| Custom operators | Supported if registered | Supported if registered | Must exist in ONNX or be implemented as custom ops in the runtime |
| Typical use | Legacy C++ deployments | New deployments inside the PyTorch ecosystem | Cross-framework, hardware-optimized inference |

**Practical notes.**

- **Graph and scatter operations** (message passing, `index_add`, sparse ops) and **FFTs** (FNO) are the usual export problems; check operator support early.
- **Always test numerical parity** on several input sizes after export, and run the regression suite on the exported artefact, not only on the training model.
- **TensorRT** (via ONNX or Torch-TensorRT) can give large speed-ups with fp16 or int8, but re-validate accuracy: reduced precision can hurt physics outputs. See [When mixed precision hurts](scaling-to-large-meshes.md#when-mixed-precision-hurts).

## Docker and Kubernetes

**Docker** packages the model, the preprocessing code and every dependency (CUDA, cuDNN, PyTorch, geometry libraries) into an image, so the service behaves identically on a laptop, the cloud and a customer's cluster. Pin versions, keep images slim (multi-stage builds), scan them for vulnerabilities, and tag them with the model and data versions.

**Kubernetes** runs those containers at scale:

- **Inference services:** deployments behind a load balancer, GPU scheduling (the NVIDIA device plugin), autoscaling on queue length or GPU utilization, rolling updates and canary releases; serving frameworks such as NVIDIA Triton, KServe, TorchServe or Ray Serve.
- **Asynchronous jobs** for large meshes: a request queue and workers, rather than synchronous HTTP calls that time out.
- **Batch training jobs** with Kubeflow, Ray or plain Jobs.

**HPC schedulers (Slurm, PBS)** remain the norm for **simulation data generation** on CPU clusters, and many teams combine them: Slurm for solvers, Kubernetes for ML services. See [Pipeline for 1,000 CFD simulations](generating-simulation-training-data.md#pipeline-for-1000-cfd-simulations).

## Experiment tracking and model versions

- **Experiment tracking** (MLflow, Weights & Biases, Neptune, ClearML): every run logs its configuration, code commit, dataset version, random seeds, metrics over time, and artefacts (checkpoints, plots, evaluation tables). Comparing runs and reproducing a result becomes routine.
- **Model registry:** each candidate model gets a version and moves through stages (candidate → staging → production → archived) after passing the regression suite; production can be rolled back instantly.
- **Model card** per version: intended use, valid input ranges (geometry families, speeds, loads), training data version, metrics overall and per slice, known failure modes, calibration results.
- **Semantic versioning for behaviour:** a version change that alters predictions on golden cases is a minor or major release, communicated to users.

## Integrating into an engineer's CAE workflow

Engineers live in CAD, pre-processors and post-processors; a surrogate that requires a new tool and new habits will not be used.

1. **Inputs they already have:** accept STEP, STL or the native mesh export; read units and coordinate frames explicitly; validate and give clear errors.
2. **Outputs they can open:** VTK/VTP/VTU or EnSight files for ParaView or the vendor post-processor, CSV or JSON for scalars, the same field names and units as the solver.
3. **Where they work:** a plugin or script inside the CAE tool (Ansys ACT/PyAnsys, Siemens NX Open, Abaqus scripting, ParaView plugins), a Python API for optimization workflows (optiSLang, modeFRONTIER, HEEDS), and a REST API for web tools.
4. **Trust signals with every prediction:** uncertainty, an "inside the validated range" flag, the nearest training designs, and one-click submission of the case to the full solver.
5. **The solver stays in the loop:** final designs are verified with the solver; the surrogate is used for exploration and screening.
6. **Speed where it matters:** interactive (seconds) for geometry edits, batch for design-of-experiments sweeps.

NVIDIA's PhysicsNeMo-CFD, for example, packages tools for evaluating and integrating pretrained models into CFD workflows; commercial platforms such as Ansys SimAI provide the whole pipeline as a product. See [Industry](../industry.md).

## Monitoring and retraining

**Monitor:**

- **Inputs:** geometry descriptors (size, bounding box, curvature statistics, latent embeddings), operating conditions, and the rate of out-of-distribution flags.
- **Outputs:** distributions of predicted quantities; sudden shifts are a warning.
- **Fallback rate:** the share of requests routed to the solver.
- **Ground truth when available:** periodically run the solver on a random sample of production requests (spot checks) and track surrogate-vs-solver error and interval coverage.
- **Operations:** latency, errors, GPU utilization, cost.
- **User feedback:** cases engineers flag as wrong.

**Retrain when** the OOD rate rises (a new design family is arriving), spot-check errors exceed thresholds, a solver or meshing change alters the "truth", or enough new verified data has accumulated. Fallback and spot-check runs are free, well-targeted training data: feed them back, as in [active learning](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions). Retraining goes through the same regression suite and registry, and the release notes say how predictions changed. See [Design: drift monitoring](design-drift-monitoring.md).

## When customers will not upload their CAD

CAD is a manufacturer's core intellectual property; many will not send it to a vendor's cloud. Options, roughly in order of how often they are used:

1. **Deploy in the customer's environment:** an on-premises installation or the customer's own cloud account (a VPC deployment), shipped as containers; the data never leaves. This is common for commercial platforms aimed at automotive and aerospace.
2. **Train and fine-tune on the customer's site:** ship the training pipeline and a pretrained base model; the customer (or forward-deployed engineers working under their security rules) fine-tunes on their simulations. Contracts define who owns the resulting weights.
3. **Share derived representations instead of geometry:** design parameters, coarse point clouds, or latent encodings computed on-site. Be honest that these can still leak information, and that is a contractual and technical question to settle.
4. **Federated learning:** train across several sites without pooling data, exchanging only model updates; helpful in principle, operationally complex, and updates can still leak information without safeguards (secure aggregation, differential privacy).
5. **Confidential computing:** processing inside hardware-isolated enclaves in the cloud; emerging for GPU workloads.
6. **Start with public or synthetic proxies:** demonstrate the approach on DrivAerML, DrivAerNet++ or SimJEB-like data, then move to the customer's data on their infrastructure.

Always: encryption at rest and in transit, access control and audit logs, data retention and deletion policies, and a clear agreement on whether the customer's data may improve the vendor's general model.

## Code: export, parity checks and a regression suite

A small point-wise surrogate (per-point pressure coefficient and a global drag value) is packaged three ways and checked against eager PyTorch on a point count it was not exported with; a regression suite then runs golden-case, invariance, resampling and sanity checks; finally the dataset manifest is hashed into a model card. Requires `pip install onnx onnxscript onnxruntime`.

```python
import hashlib, json, os, tempfile, warnings
import torch, torch.nn as nn
import onnxruntime as ort
warnings.filterwarnings("ignore")

class PointSurrogate(nn.Module):
    """Per point (x, y, z, nx, ny, nz) -> pressure coefficient, plus a global drag head."""
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.cp = nn.Linear(128, 1)
        self.drag = nn.Linear(64, 1)
    def forward(self, pts):                                   # pts: (batch, N, 6)
        h = self.point(pts)
        g = h.max(dim=1, keepdim=True).values                 # global feature by max pooling
        cp = self.cp(torch.cat([h, g.expand_as(h)], -1)).squeeze(-1)
        return cp, self.drag(g.squeeze(1)).squeeze(-1)

torch.manual_seed(0)
model = PointSurrogate().eval()
example = torch.randn(1, 2048, 6)
work = tempfile.mkdtemp()

# 1. Package three ways, then check parity with eager PyTorch on a different number of points.
scripted = torch.jit.trace(model, example)
exported = torch.export.export(model, (example,), dynamic_shapes=({1: torch.export.Dim("n_points", max=10**6)},))
onnx_path = os.path.join(work, "surrogate.onnx")
torch.onnx.export(model, (example,), onnx_path, input_names=["points"], output_names=["cp", "drag"],
                  dynamic_axes={"points": {1: "n_points"}, "cp": {1: "n_points"}}, verbose=False)
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

test = torch.randn(1, 5000, 6)
with torch.no_grad():
    ref_cp, ref_drag = model(test)
    outs = {"TorchScript (trace)": scripted(test), "torch.export": exported.module()(test)}
outs["ONNX Runtime"] = [torch.from_numpy(a) for a in session.run(None, {"points": test.numpy()})]
for name, (cp, drag) in outs.items():
    print(f"{name:<20} max |cp diff| {(cp - ref_cp).abs().max().item():.1e}, |drag diff| {(drag - ref_drag).abs().max().item():.1e}")

# 2. A regression suite run on every candidate model.
def regression_suite(model, golden_inputs, golden_cp, tol=0.02):
    report = {}
    with torch.no_grad():
        cp, drag = model(golden_inputs)
        rel = ((cp - golden_cp).norm(dim=1) / golden_cp.norm(dim=1)).max().item()
        report["golden cases (max rel. L2 vs reference)"] = (rel, rel <= tol)
        perm = torch.randperm(golden_inputs.shape[1])
        cp_p, drag_p = model(golden_inputs[:, perm])
        d = max((cp_p - cp[:, perm]).abs().max().item(), (drag_p - drag).abs().max().item())
        report["point-order invariance"] = (d, d <= 1e-5)
        _, drag_half = model(golden_inputs[:, ::2])            # the same surface sampled with half the points
        d = ((drag_half - drag).abs() / drag.abs().clamp(min=1e-6)).max().item()
        report["resampling sensitivity of drag"] = (d, d <= 0.05)
        report["finite outputs"] = (float(torch.isfinite(cp).all()), bool(torch.isfinite(cp).all()))
    return report

golden_inputs = torch.randn(4, 4000, 6)
with torch.no_grad():
    golden_cp = model(golden_inputs)[0] * 1.01                 # stand-in for stored reference solver results
for check, (value, passed) in regression_suite(model, golden_inputs, golden_cp).items():
    print(f"{'PASS' if passed else 'FAIL'}  {check}: {value:.3g}")

# 3. Version data and model together: a content hash of the dataset manifest goes into the model card.
manifest = [{"case": f"design_{i:03d}", "solver": "solverX 2.1", "mesh": "fine-v3",
             "file_sha256": hashlib.sha256(f"{i}".encode()).hexdigest()} for i in range(500)]
dataset_version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
model_card = {"model": "point-surrogate", "version": "1.4.0", "dataset_version": dataset_version,
              "git_commit": "abc1234", "test_rel_l2": 0.031, "valid_ranges": {"speed_m_s": [20, 40]}}
print("model card:", json.dumps(model_card))
```

Output:

```
TorchScript (trace)  max |cp diff| 0.0e+00, |drag diff| 0.0e+00
torch.export         max |cp diff| 0.0e+00, |drag diff| 0.0e+00
ONNX Runtime         max |cp diff| 1.2e-07, |drag diff| 1.5e-08
PASS  golden cases (max rel. L2 vs reference): 0.0099
PASS  point-order invariance: 0
FAIL  resampling sensitivity of drag: 0.16
PASS  finite outputs: 1
model card: {"model": "point-surrogate", "version": "1.4.0", "dataset_version": "a9e0226087f9", "git_commit": "abc1234", "test_rel_l2": 0.031, "valid_ranges": {"speed_m_s": [20, 40]}}
```

**What it shows.** All three packaged forms reproduce the eager model on 5,000 points although they were exported with 2,048 (exactly for TorchScript and torch.export, to about 10⁻⁷ for ONNX Runtime). The suite passes the golden cases and the point-order test, but **fails the resampling test**: the max-pooled drag head changes by 16% when the same surface is sampled with half as many points, exactly the kind of mesh dependence that would surprise a customer who remeshes their part. A suite that never fails is not testing enough. The model card ties this model version to one exact dataset version.

## In an interview

**Thirty-second answer to "What goes into a regression test suite for a surrogate?"** "Golden cases with reference solver results and thresholds, per slice as well as overall, including integrated quantities like drag. Invariance tests: point order, remeshing, units, rigid motions where the physics allows. Physics and sanity checks: finite values, physical ranges, conservation. Out-of-distribution inputs must trigger the flag and the solver fallback. Calibration of the uncertainty, parity between the exported artefact and the training model, and latency and memory budgets. It runs in CI on every candidate before the registry promotes it."

**Likely follow-ups**

- *ONNX or torch.export?* torch.export for new deployments inside the PyTorch ecosystem; ONNX for cross-runtime, hardware-optimized serving; check operator support for scatter and FFT ops, and always test parity.
- *How do you version data?* An immutable dataset identifier (content hash of a manifest with solver versions, settings and checksums), recorded in every model's card.
- *The customer will not upload CAD. What now?* Deploy in their environment and fine-tune there, or work from agreed derived representations, with contracts on weight ownership.

## References

- [PyTorch: torch.export documentation](https://docs.pytorch.org/docs/stable/export.html) and [ONNX export](https://docs.pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime](https://onnxruntime.ai/) and [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [MLflow](https://mlflow.org/) and [DVC](https://dvc.org/)
- [Hidden Technical Debt in Machine Learning Systems (Sculley et al., NeurIPS 2015)](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993)
- [PhysicsNeMo-CFD](https://github.com/NVIDIA/physicsnemo-cfd) and [PySimAI documentation](https://simai.docs.pyansys.com)
