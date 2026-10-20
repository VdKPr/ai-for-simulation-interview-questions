# Design: Heat Exchanger Digital Twin

[← Questions](../README.md) · [All answers](README.md) · [System design framework](system-design-framework.md)

**Question.** Design a real-time digital twin of a heat exchanger that combines sensor data with simulation.

**Short answer.**

- **What the twin does:** it estimates the exchanger's full state (outlet temperatures, the internal temperature field, heat duty) from a few sensors in real time. It also tracks slowly changing parameters, above all the **fouling resistance**, and forecasts when performance will cross a maintenance threshold.
- **The model:** a fast physics-based or surrogate model, such as an ε-NTU model, a reduced-order model of CFD/CHT, or a neural surrogate trained on simulations. It is **corrected online by data assimilation**: a Kalman filter (an extended or ensemble variant) estimates both state and parameters from the sensor stream.
- **Engineering details:**
  - sensor validation (drift, dropouts);
  - uncertainty on every estimate;
  - checks that estimated parameters stay physical;
  - a fallback to a simple model when the surrogate is outside its training range.

**On this page**

- [Requirements and clarifying questions](#requirements-and-clarifying-questions)
- [Physics and the model hierarchy](#physics-and-the-model-hierarchy)
- [Combining sensors and model: data assimilation](#combining-sensors-and-model-data-assimilation)
- [Estimating fouling](#estimating-fouling)
- [Sensor data quality](#sensor-data-quality)
- [Architecture](#architecture)
- [Risks and roadmap](#risks-and-roadmap)
- [In an interview](#in-an-interview)

## Requirements and clarifying questions

- **Purpose:** monitoring (current performance), prediction (when to clean), control (setpoints), or what-if analysis? Each needs different outputs and latency.
- **Sensors:** inlet and outlet temperatures and flow rates on both sides, pressure drops; sampling rate; reliability.
- **Type of exchanger:** shell-and-tube, plate, fin-tube; single-phase or condensing (two-phase is much harder).
- **Latency:** seconds for monitoring; control may need sub-second.
- **Available simulations:** CFD/CHT of the design; design data sheets; historical operating data.
- **Where it runs:** at the edge (plant computer) or in the cloud; connectivity.

## Physics and the model hierarchy

**Overall balance.** The heat duty from each side must agree: Q = ṁ_h c_p,h (T_h,in − T_h,out) = ṁ_c c_p,c (T_c,out − T_c,in), and Q = U A ΔT_lm. With **ε-NTU**, outlet temperatures follow from inlet conditions, flow rates and UA, in microseconds. Fouling adds a resistance: 1/U = 1/U_clean + R_f ([Heat transfer basics](heat-transfer-basics.md)).

| Model | Speed | What it gives | Role |
| --- | --- | --- | --- |
| ε-NTU / lumped model | Microseconds | Outlet temperatures, duty | Always-available baseline and fallback |
| 1D segmented model (along the flow path) | Milliseconds | Temperature profiles, local fouling | Good default for twins |
| Reduced-order model from CFD/CHT (POD) | Milliseconds | Approximate fields | When internal fields matter ([Reduced-order models](reduced-order-models.md)) |
| Neural surrogate trained on CFD/CHT | Milliseconds | Full fields, non-linear effects (maldistribution) | When conditions vary widely and fields are needed |
| Full CFD/CHT | Hours | Reference | Offline: training data and verification |

Start with the simplest model that answers the question. Many twins need only the 1D model plus fouling estimation.

## Combining sensors and model: data assimilation

The model has errors and unknown parameters; sensors are sparse and noisy. **Data assimilation** combines them:

- **State vector:** temperatures (or the surrogate's latent state) plus slowly varying parameters (fouling resistance, a heat-transfer correlation factor).
- **Forecast step:** the model advances the state with current inputs (inlet temperatures, flows).
- **Update step:** compare predicted and measured outputs; correct the state and parameters in proportion to their uncertainties (the Kalman gain).
- **Variants:** an extended Kalman filter for mildly nonlinear models; an **ensemble Kalman filter** for nonlinear or neural models (no Jacobians needed, works with any black-box surrogate); a particle filter for strongly non-Gaussian cases; moving-horizon estimation if constraints matter.

The result: estimates of unmeasured quantities with uncertainty, and parameters that track the real exchanger. See [Digital twins: simulation plus measurements](multi-fidelity-learning.md#digital-twins-simulation-plus-measurements).

## Estimating fouling

- Fouling resistance R_f grows slowly (days to months) and cannot be measured directly; it is the main thing operators want to know.
- Treat R_f as a parameter in the state vector with a **random-walk model** (it can change slowly); the filter estimates it from the gap between predicted and measured outlet temperatures.
- **Identifiability:** if flow rates barely change, fouling and a correlation error look the same. Excitation (operating changes) helps; prior knowledge of the clean state (commissioning data) anchors it.
- **Forecast:** fit a fouling growth model (linear, asymptotic) to the estimated R_f history; predict when performance crosses a threshold, with uncertainty; recommend cleaning schedules.

## Sensor data quality

Real sensors fail in ways the twin must survive:

- **Range and rate checks;** flat-line and spike detection.
- **Physical consistency:** energy balance between the two sides; a large imbalance points to a faulty sensor or flow meter.
- **Drift:** compare redundant sensors; the filter's innovation statistics (measured minus predicted) reveal slow sensor drift.
- **Missing data:** the filter continues with forecasts only; uncertainty grows, which the dashboard shows.
- **Time alignment:** sensors on different clocks or sampling rates.

## Architecture

```
plant sensors ─► ingest + validation (range, balance, drift) ─► data assimilation (EnKF) ─► state + parameters (± σ)
                                                                     ▲        │
                                       fast model (1D / ROM / NN) ───┘        ├─► dashboard: duty, outlet T, fouling trend
                                                                              ├─► forecast: cleaning date ± uncertainty
       offline: CFD/CHT ─► training data ─► surrogate ─► validation           └─► alerts (sensor fault, abnormal fouling)
```

Runs at the edge for latency and resilience, with periodic sync to the cloud for retraining and fleet comparisons.

## Risks and roadmap

- **Model–plant mismatch** beyond what parameters can absorb (bypass flows, maldistribution): watch the innovation statistics; fall back to the simple model and flag it.
- **Surrogate out of range:** operating points outside the training data; check inputs against training ranges ([OOD detection](out-of-distribution-and-generalization.md)).
- **Unidentifiable parameters:** keep the parameter set small.
- **v0:** ε-NTU or 1D model with an EKF estimating fouling; dashboard.
- **v1:** neural or ROM surrogate from CFD/CHT for internal fields; ensemble Kalman filter.
- **v2:** predictive maintenance and optimization of cleaning schedules across a fleet.

## In an interview

**Thirty-second answer.** "I'd first ask what the twin is for. For monitoring and maintenance, the key unknown is fouling. I'd start with a fast physical model, a segmented ε-NTU model, and use CFD only offline to build a surrogate if internal fields matter. An ensemble Kalman filter combines the model with the inlet and outlet sensors, estimating the temperatures and the fouling resistance with uncertainty, and the fouling trend forecasts when to clean. Sensor validation, especially energy-balance checks and drift detection, is as important as the model, and the twin falls back to the simple model when the surrogate is outside its range."

**Likely follow-ups**

- *Why a Kalman filter rather than retraining the model on sensor data?* Sensors are sparse and continuous; filtering updates in real time with uncertainty, without retraining.
- *How do you validate the twin?* Hold out a sensor and predict it; compare fouling estimates with inspection or cleaning records.

## References

- G. Evensen, *Data Assimilation: The Ensemble Kalman Filter*, 2nd ed., Springer, 2009.
- F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, *Fundamentals of Heat and Mass Transfer*, Wiley (ε-NTU method).
- [Heat transfer basics](heat-transfer-basics.md), [Multi-fidelity learning](multi-fidelity-learning.md), [Reduced-order models](reduced-order-models.md)
