# Design: Simulation Data Pipeline

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design the pipeline that generates and stores 10,000 CFD simulations for training.

**Short answer.**

- **Treat it as a production system, not a script.** A space-filling design of experiments produces parameter sets. Geometry is generated and checked automatically, and meshing and solving run as **idempotent, containerized jobs** on a cluster or in the cloud.
- **Every run gets automated quality checks:** convergence, mass balance, y+, mesh quality and outlier quantities of interest. Failures are diagnosed and recorded, never silently dropped, because failures are themselves data about where the design space is hard.
- **Storage:**
  - raw solver output in cheap object storage;
  - ML-ready, chunked data (Zarr or HDF5) containing only the fields and resolution the model needs;
  - a metadata catalogue (parameters, solver version, settings, check results) in a database.
- **Versioned datasets,** so every model can be traced back to the exact data it used.
- **Do the arithmetic first:** 10,000 steady RANS runs can mean millions of core-hours and terabytes of output (and far more storage for unsteady runs that save many time steps), so decide what to keep before starting.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Budget arithmetic](#budget-arithmetic)
- [Pipeline stages](#pipeline-stages)
- [Quality checks](#quality-checks)
- [Storage layout](#storage-layout)
- [Orchestration and failure handling](#orchestration-and-failure-handling)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **What is the dataset for?** A specific surrogate (which outputs, which resolution) or a general pretraining corpus? This decides what to store.
- **Design space:** how many parameters, which ranges; parametric morphing or a geometry generator; which operating conditions.
- **Fidelity and cost per run:** steady RANS on a few million cells, or unsteady scale-resolving runs; core-hours per run.
- **Compute:** on-premises cluster, cloud, licences (commercial solvers are often limited by licence count, not hardware).
- **Timeline and budget.**
- **Ownership and licences** of geometry and results; who may use them for training.

## Budget arithmetic

Rough numbers decide the design. The block below assumes a steady RANS case with 10 million cells.

```python
runs = 10_000
cells = 10e6
core_hours_per_run = 400                 # steady RANS, ~10M cells (order of magnitude; measure on a pilot)
cores_available = 5_000
failure_rate = 0.08                      # fraction needing a re-run
total_core_hours = runs * core_hours_per_run * (1 + failure_rate)
wall_days = total_core_hours / cores_available / 24

bytes_per_value = 4
volume_fields = 7                        # p, u, v, w, nut, k, omega
surface_points = 0.5e6
surface_fields = 4                       # p, tau_x, tau_y, tau_z
raw_gb = cells * volume_fields * bytes_per_value * 1.5 / 1e9          # x1.5 for mesh and overhead
ml_surface_gb = surface_points * (3 + 3 + surface_fields) * bytes_per_value / 1e9   # coords, normals, fields
ml_volume_gb = 0.5e6 * (3 + volume_fields) * bytes_per_value / 1e9    # 0.5M subsampled volume points

print(f"compute: {total_core_hours / 1e6:.2f} M core-hours, about {wall_days:.0f} days on {cores_available:,} cores")
print(f"raw volume output: {raw_gb:.2f} GB per run, {runs * raw_gb / 1e3:.0f} TB in total")
print(f"ML-ready (surface + 0.5M volume points): {1e3 * (ml_surface_gb + ml_volume_gb):.0f} MB per run, "
      f"{runs * (ml_surface_gb + ml_volume_gb) / 1e3:.1f} TB in total")
```

Output:

```
compute: 4.32 M core-hours, about 36 days on 5,000 cores
raw volume output: 0.42 GB per run, 4 TB in total
ML-ready (surface + 0.5M volume points): 40 MB per run, 0.4 TB in total
```

The conclusion is typical: the ML-ready data is a small fraction of the raw output. Raw files go to cold storage, or are not kept beyond the fields needed. Always run a **pilot of 20–50 cases** to measure real cost, failure rate and file sizes before launching the campaign.

## Pipeline stages

1. **Design of experiments:** Latin hypercube or Sobol sampling over the parameters, possibly in batches so later batches can be placed by active learning ([Design of experiments](design-of-experiments-and-active-learning.md)). Each sample gets a stable ID.
2. **Geometry:** parametric CAD or morphing. Check validity (watertight, no self-intersections, feasible dimensions) and record infeasible samples.
3. **Meshing:** a scripted mesher with fixed settings per family (refinement zones, prism layers). Check mesh quality (skewness, non-orthogonality, aspect ratio) and reject or remesh ([Mesh quality](meshing-and-mesh-quality.md)).
4. **Solve:** containerized solver with pinned versions and settings files under version control. Use the same numerics for every run; changing solver settings mid-campaign creates hidden label shifts.
5. **Post-process:** extract quantities of interest and fields; interpolate to the ML representation (surface points, subsampled volume points); non-dimensionalize.
6. **Quality checks** (below), then **register** the run in the catalogue with its check results.
7. **Dataset release:** freeze a version (a list of run IDs plus their processing version) for training, with a datasheet: ranges, failures, known issues.

See also [Pipeline for 1,000 CFD simulations](generating-simulation-training-data.md#pipeline-for-1000-cfd-simulations).

## Quality checks

| Check | How | Action on failure |
| --- | --- | --- |
| Residual convergence | Final residuals below threshold | Re-run with more iterations or lower relaxation |
| QoI convergence | Drag and lift history flat over the last N iterations (or statistically steady for unsteady runs) | Extend run; flag oscillatory cases (possibly physical unsteadiness) |
| Conservation | Mass imbalance between inlet and outlet | Investigate mesh or set-up |
| Wall resolution | y+ distribution in the range the wall treatment needs | Remesh ([Boundary layers](boundary-layers-and-wall-treatment.md)) |
| Physical plausibility | QoIs within expected ranges; no NaNs; positive turbulence quantities | Inspect; quarantine |
| Statistical outliers | QoI far from neighbours in parameter space (e.g. via a GP fitted to the dataset) | Manual review: an error or real physics (separation)? |
| Mesh sensitivity | A few cases on a refined mesh | Estimate the label noise floor ([Mesh convergence](verification-validation-and-mesh-convergence.md)) |

Record **why** runs failed. A cluster of failures in one region of the design space often marks a physical change (separation, stall) that the model will also struggle with.

## Storage layout

- **Raw:** solver case directories, compressed, in object storage (cold tier), keyed by run ID. Keep the inputs (geometry, mesh settings, solver set-up) even if outputs are deleted: they let you regenerate.
- **ML-ready:** one Zarr group (or HDF5 file) per run with arrays for coordinates, normals, fields and connectivity, chunked by points so training can read random subsets; float32 (or float16 for storage, with care) ([File formats](simulation-file-formats.md)).
- **Catalogue:** a table (PostgreSQL or Parquet) with run ID, parameters, solver and pipeline versions, check results, QoIs and file locations. This is what people query to build training sets.
- **Dataset versions:** immutable manifests (run IDs plus processing version) tracked with DVC or a lakehouse table format, so every model is reproducible ([Versioning data together with models](production-and-deployment.md#versioning-data-together-with-models)).

## Orchestration and failure handling

- **Workflow engine:** Airflow, Prefect, Nextflow or Argo, on top of Slurm or Kubernetes; each stage is an idempotent task keyed by run ID, so re-running is safe.
- **Retries** with backoff for infrastructure failures; **automatic fixes** for known numerical failures (lower relaxation, a smaller time step); **quarantine** after N attempts.
- **Licence-aware scheduling** for commercial solvers.
- **Monitoring dashboard:** runs queued, running, passed, failed by reason; cost so far; estimated completion.
- **Spot or pre-emptible instances** with checkpointing for long runs to cut cloud cost.

## Architecture

```
DoE sampler ─► geometry gen + checks ─► mesher + quality ─► solver (container) ─► post-process ─► QC checks
     ▲                                                                                              │
     │                       failures (reason) ◄──────────────────── fail ─────────────────────────┤
     │                                                                                              │ pass
active learning (later batches) ◄── model trained on dataset version ◄── dataset manifest ◄── catalogue
                                                                                                    │
                          raw case files ─► object storage (cold)       ML-ready Zarr ─► object storage (hot)
```

## Risks and roadmap

- **Hidden inconsistency:** solver version or settings changing mid-campaign. Pin everything; record versions per run.
- **Design-space holes:** many failures in one region. Report them; do not let the dataset quietly exclude the hard cases.
- **Cost overrun:** measured in the pilot, re-estimated weekly.
- **v0:** a pilot of 50 runs end to end, with checks and the catalogue.
- **v1:** the full campaign in batches of 1,000, with a model trained after each batch to monitor the learning curve; stop early if accuracy saturates.
- **v2:** active-learning batches; multi-fidelity runs.

## In an interview

**Thirty-second answer.** "I'd start with the arithmetic from a pilot of about 50 runs: core-hours, failure rate and file sizes. That usually shows that you can't keep everything, so you store only what the model needs in ML-ready form. The pipeline is DoE, then geometry, meshing and solving as containerized, idempotent jobs with pinned versions. Every run gets automated checks on convergence, conservation, y+ and outliers, and failures are recorded with reasons. Raw files go to cold storage, chunked Zarr goes to hot storage, a catalogue holds the metadata, and dataset versions are immutable so every model is reproducible. I'd run in batches and watch the learning curve to decide whether all 10,000 runs are needed."

**Likely follow-ups**

- *How do you detect a bad simulation that converged?* Outlier checks against neighbours in parameter space, conservation checks and spot mesh-refinement studies.
- *How do you feed this to training?* Chunked reads, a data loader that samples point subsets, prefetching ([Feeding terabytes to the GPUs](scaling-to-large-meshes.md#feeding-terabytes-to-the-gpus)).

## References

- [Generating simulation training data](generating-simulation-training-data.md), [Simulation file formats](simulation-file-formats.md)
- [Zarr](https://zarr.readthedocs.io/), [DVC](https://dvc.org/doc), [Prefect](https://docs.prefect.io/), [Apache Airflow](https://airflow.apache.org/docs/)
- [DrivAerML](https://arxiv.org/abs/2408.11969) (an example of a large, documented CFD dataset release)
