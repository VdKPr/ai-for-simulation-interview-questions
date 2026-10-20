# Design: Surrogate Serving System

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a serving system for 1,000 engineers using a surrogate interactively.

**Short answer.**

- **Estimate the load first.** 1,000 engineers do not all query at once. Interactive use means a few requests per second at peak, with short bursts above that, each needing geometry preprocessing (CPU) and one or a few model passes (GPU). Size the GPU pool from **peak** concurrent requests and per-request GPU time, with headroom, using autoscaling.
- **Stages:**
  - an API gateway with authentication;
  - an asynchronous job queue for preprocessing and inference;
  - CPU workers for geometry;
  - GPU inference servers with dynamic batching (Triton or TorchServe; an ONNX or `torch.export` model);
  - an object store for results (fields are large, so return a URL rather than an inline payload);
  - a cache keyed on a geometry hash and the parameters.
- **Engineering concerns:**
  - latency targets per stage;
  - multiple model versions with routing, canaries and rollback;
  - tenancy and access control on CAD;
  - observability (latency, errors, GPU use, OOD-flag rate);
  - a solver fallback queue for flagged cases.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Capacity estimate](#capacity-estimate)
- [Request path](#request-path)
- [Inference servers and batching](#inference-servers-and-batching)
- [Caching, storage and results](#caching-storage-and-results)
- [Versions, rollout and reliability](#versions-rollout-and-reliability)
- [Security and tenancy](#security-and-tenancy)
- [Architecture](#architecture)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **Usage pattern:** how many queries per engineer per day; peaks (working hours across time zones, design reviews); batch sweeps vs single interactive queries.
- **Latency target:** for example, p95 under 5 seconds from upload to result for interactive use; sweeps can be slower.
- **Payload:** input geometry size (an STL of tens of MB), output fields (millions of points × several channels).
- **Models:** one or many (per product line, per customer); how often they change.
- **Deployment:** customer cloud, on-premises GPU cluster, or vendor SaaS.
- **Availability:** business hours or 24/7; what happens during outages (queue, or degrade to the solver).

## Capacity estimate

The block below works through the numbers for the peak hour.

```python
import math
engineers = 1_000
active_fraction_peak = 0.3            # engineers active in the busiest hour
queries_per_active_hour = 20          # interactive exploration: one query every 3 minutes
peak_qps = engineers * active_fraction_peak * queries_per_active_hour / 3600
burst_factor = 3                      # short bursts above the hourly average
gpu_seconds_per_query = 0.4           # model inference on one GPU (measure it)
cpu_seconds_per_query = 2.0           # geometry checks, sampling, post-processing
target_utilization = 0.6              # headroom keeps queueing delay low

gpus = math.ceil(peak_qps * burst_factor * gpu_seconds_per_query / target_utilization)
cpu_cores = math.ceil(peak_qps * burst_factor * cpu_seconds_per_query / target_utilization)
fields_mb = 2e6 * 4 * 4 / 1e6         # 2M surface points x 4 channels x float32
daily_queries = engineers * 0.6 * 40
print(f"peak load {peak_qps:.1f} queries/s (bursts {peak_qps * burst_factor:.0f}/s) -> about {gpus} GPUs and "
      f"{cpu_cores} CPU cores for preprocessing")
print(f"result fields {fields_mb:.0f} MB per query; {daily_queries:,.0f} queries/day -> "
      f"{daily_queries * fields_mb / 1e6:.2f} TB/day if every result is stored")
```

Output:

```
peak load 1.7 queries/s (bursts 5/s) -> about 4 GPUs and 17 CPU cores for preprocessing
result fields 32 MB per query; 24,000 queries/day -> 0.77 TB/day if every result is stored
```

Two conclusions are typical. GPU count is modest, and preprocessing (CPU) often dominates, so profile the whole path, not just the model. Storing every full-field result is expensive, so keep results for a limited time, compress them, or store only quantities of interest plus a re-computable reference (geometry hash, model version).

Queueing intuition: as utilization approaches 100%, waiting time grows sharply (for an M/M/1 queue, the mean time in system is the service time divided by 1 − utilization). Target 50–70% at peak.

## Request path

1. **Client** (CAD plug-in, web app or Python SDK) uploads geometry to object storage with a pre-signed URL; submits a job (geometry reference, parameters, model version).
2. **API gateway:** authentication, quotas per team, validation of parameters.
3. **Cache lookup:** hash of geometry plus parameters plus model version. A hit returns immediately (common when engineers revisit designs).
4. **Queue** (Redis, RabbitMQ, Kafka or a cloud queue) decouples submission from processing and absorbs bursts.
5. **CPU workers:** geometry validation, units, sampling ([Mesh data processing](mesh-data-processing.md)).
6. **GPU inference:** model ensemble; uncertainty; OOD score.
7. **Post-processing:** integrated quantities; files (VTP) to object storage.
8. **Response:** status via polling or WebSocket; result URL; QoIs with intervals and the OOD flag; solver job queued if flagged.

Synchronous calls are possible for small, fast models; the asynchronous job pattern is more robust for anything over a second.

## Inference servers and batching

- **Model format:** exported with ONNX or `torch.export`, with parity tests against the training model ([ONNX vs TorchScript vs torch.export](production-and-deployment.md#onnx-vs-torchscript-vs-torchexport)).
- **Server:** NVIDIA Triton or TorchServe; **dynamic batching** combines concurrent requests (pad or bucket by point count, because geometries have different sizes).
- **Large inputs:** chunk query points so memory is bounded; share the geometry encoding across chunks.
- **Precision:** fp16 or bf16 inference where validated ([When mixed precision hurts](scaling-to-large-meshes.md#when-mixed-precision-hurts)).
- **Ensembles:** run members in one batched pass, or on separate GPUs in parallel.
- **Autoscaling** on queue length and GPU utilization (Kubernetes with a GPU node pool); keep a warm minimum to avoid cold starts at 9 a.m. ([Docker and Kubernetes](production-and-deployment.md#docker-and-kubernetes)).

## Caching, storage and results

- **Cache:** results keyed by (geometry hash, parameters, model version); invalidated automatically by a new model version.
- **Object storage** for geometry and results with lifecycle rules (delete or archive after N days).
- **Metadata database:** jobs, users, versions, QoIs, flags; this is also the monitoring and audit record.
- **Result formats:** VTP or similar for ParaView; QoIs as JSON; thumbnails for the UI.

## Versions, rollout and reliability

- **Model registry** with versions and their validation reports; each request records the version used ([Experiment tracking and model versions](production-and-deployment.md#experiment-tracking-and-model-versions)).
- **Pinning:** projects can pin a model version so results stay reproducible during a design programme.
- **Rollout:** shadow traffic for a new version, then a canary percentage, then full; automatic rollback on error or regression ([A regression test suite](production-and-deployment.md#a-regression-test-suite-for-a-surrogate)).
- **Reliability:** retries for transient failures; dead-letter queue; health checks; graceful degradation (queue jobs during GPU outages; route to the solver if needed).
- **Observability:** latency per stage (p50, p95, p99), queue depth, GPU utilization, error rates, OOD-flag rate, cache hit rate ([Monitoring and retraining](production-and-deployment.md#monitoring-and-retraining), [Drift monitoring](design-drift-monitoring.md)).

## Security and tenancy

- CAD is sensitive: encryption at rest and in transit; access control by project; audit logs of who accessed what.
- Per-customer isolation (separate namespaces, buckets, keys) in multi-tenant deployments, or on-premises deployment for customers who require it ([When customers will not upload their CAD](production-and-deployment.md#when-customers-will-not-upload-their-cad)).
- Rate limits and quotas so one batch sweep does not starve interactive users (separate queues and priorities for interactive and batch).

## Architecture

```
CAD plug-in / web / SDK ─► API gateway (auth, quotas) ─► cache? ── hit ──────────────────────────► result
                                   │ miss
                                   ▼
                     job queue (interactive | batch priority)
                                   │
            CPU workers: validate, units, sample ─► GPU inference (Triton, dynamic batching, ensemble)
                                                         │
                       QoIs ± σ, OOD flag, fields ─► object store + metadata DB ─► result URL
                                                         │ flagged
                                                         ▼
                                                   solver queue
   model registry ─► canary / pinned versions        metrics: latency, GPU, queue, OOD rate, cache hits
```

## In an interview

**Thirty-second answer.** "I'd start with a capacity estimate: 1,000 engineers at peak means a few queries per second with bursts, so a handful of GPUs, and preprocessing on CPUs is often the real bottleneck. Requests go through an authenticated gateway to an asynchronous queue with separate priorities for interactive and batch work. CPU workers handle geometry, GPU servers like Triton run the exported model with dynamic batching, results go to object storage, and a cache keyed on a geometry hash plus model version absorbs repeats. Model versions live in a registry with canary rollout and pinning, and I'd monitor latency per stage, GPU use and the OOD-flag rate."

**Likely follow-ups**

- *How do you keep p95 latency low at peak?* Headroom (60% utilization), warm capacity, priority queues, batching with a maximum delay.
- *How do you serve many customer-specific models?* Load models on demand with an LRU cache of weights per GPU, or multi-model servers.

## References

- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server), [TorchServe](https://pytorch.org/serve/)
- M. Kleppmann, *Designing Data-Intensive Applications*, O'Reilly, 2017.
- [Production and deployment](production-and-deployment.md), [Scaling to large meshes](scaling-to-large-meshes.md)
