# Von Mises Stress, Yield and Safety Factors

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A material test gives one number, the yield strength σ_y in uniaxial tension, but a real part has a three-dimensional stress state with six components. A **yield criterion** reduces the stress state to one equivalent stress that can be compared with σ_y. The **von Mises stress** does this for ductile metals: it measures the *distortional* (shape-changing) part of the stress and ignores the hydrostatic (pressure) part, which matches how ductile metals actually yield, by shear-driven slip. Yield is predicted when σ_vm reaches σ_y. A **safety factor** is the ratio of what the part can withstand to what it carries, for example σ_y / σ_vm,max. Predicting a safety factor with a surrogate is harder than it looks: the decision depends on the *peak* stress at one location, which smooth models under-predict, which can be corrupted by singularities in the training labels, and which turns small stress errors into large errors near the pass/fail threshold.

**On this page**

- [Why a yield criterion is needed](#why-a-yield-criterion-is-needed)
- [The von Mises stress](#the-von-mises-stress)
- [Why von Mises suits ductile metals](#why-von-mises-suits-ductile-metals)
- [Tresca and other criteria](#tresca-and-other-criteria)
- [Worked examples](#worked-examples)
- [Safety factor](#safety-factor)
- [Predicting a safety factor with a surrogate](#predicting-a-safety-factor-with-a-surrogate)
- [Code: von Mises stress and a safety-factor check](#code-von-mises-stress-and-a-safety-factor-check)
- [In an interview](#in-an-interview)

## Why a yield criterion is needed

A tensile test applies stress in one direction and measures when the material starts to deform permanently. A shaft under bending and torsion, a pressure vessel or a bracket near a bolt hole has normal and shear stresses in several directions at once. Which combination makes it yield? A yield criterion answers that by defining an equivalent stress that equals the uniaxial stress in a tensile test and can be compared with σ_y for any stress state.

## The von Mises stress

In terms of the principal stresses σ₁, σ₂, σ₃:

```math
\sigma_{\text{vm}} = \sqrt{\tfrac{1}{2}\left[(\sigma_1 - \sigma_2)^2 + (\sigma_2 - \sigma_3)^2 + (\sigma_3 - \sigma_1)^2\right]}
```

In terms of the components in any coordinate system:

```math
\sigma_{\text{vm}} = \sqrt{\tfrac{1}{2}\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2 + (\sigma_{zz}-\sigma_{xx})^2\right] + 3\left(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{zx}^2\right)}
```

Equivalently, σ_vm = √(3 J₂), where J₂ is the second invariant of the **deviatoric stress** s = σ − (tr σ / 3) I, the part of the stress left after removing the average (hydrostatic) pressure.

**Properties worth remembering:**

- In uniaxial tension (only σ₁ ≠ 0), σ_vm = σ₁, so it matches the tensile test by construction.
- It is **invariant**: rotating the coordinate system does not change it.
- It is **always non-negative**: it does not distinguish tension from compression.
- Adding any hydrostatic pressure (the same value to σ₁, σ₂ and σ₃) leaves it **unchanged**.

## Why von Mises suits ductile metals

**Physics.** Metals yield by dislocations gliding on crystal planes, which is driven by shear. A purely hydrostatic stress, equal pressure in every direction, produces no shear on any plane, so it does not cause yielding: experiments on metals under very high hydrostatic pressure confirm that it barely affects yield. The von Mises stress depends only on the deviatoric part of the stress, exactly the part that drives slip.

**Energy view.** The strain energy stored in a stressed element splits into a volume-change part and a shape-change (distortion) part. The von Mises criterion says yield happens when the **distortion energy** reaches the value it has at yield in a tensile test. This is why it is also called the *maximum distortion energy criterion*.

**Experiments.** For most ductile metals, measured yield under combined loading lies close to the von Mises prediction, slightly better than Tresca.

## Tresca and other criteria

| Criterion | Equivalent stress | Good for | Notes |
| --- | --- | --- | --- |
| **Von Mises** | √(3J₂), above | Ductile metals | Smooth; standard in FEA post-processing |
| **Tresca** (maximum shear) | σ₁ − σ₃ (yield when the maximum shear stress reaches σ_y/2) | Ductile metals, conservative design | Up to about 15% more conservative than von Mises (the factor 2/√3 ≈ 1.155 in pure shear) |
| **Maximum principal stress** (Rankine) | σ₁ | Brittle materials: cast iron, ceramics, glass | Brittle materials fail in tension at a crack, not by shear |
| **Mohr-Coulomb, Drucker-Prager** | Depend on pressure | Soils, rock, concrete, some polymers | Strength rises with confining pressure |
| **Tsai-Wu, Hashin** and others | Direction-dependent | Fibre composites | Separate fibre and matrix failure modes |

**When von Mises is the wrong number:**

- **Brittle materials:** use principal stresses.
- **Fatigue:** the sign of the stress matters (compression closes cracks, tension opens them), but von Mises is always positive. Fatigue assessments use principal stresses, a signed von Mises, or critical-plane methods. See [Fatigue](fatigue.md).
- **Buckling:** a part can collapse at stresses far below yield. See [Buckling](buckling.md).
- **Above yield in a linear analysis:** the number is not a real stress; the material would have redistributed it. See [When linear elasticity is not enough](stress-strain-and-linear-elasticity.md#when-linear-elasticity-is-not-enough).

## Worked examples

**Pure shear** (a shaft in torsion): σ₁ = τ, σ₂ = 0, σ₃ = −τ. Then

```math
\sigma_{\text{vm}} = \sqrt{\tfrac{1}{2}\left[\tau^2 + \tau^2 + 4\tau^2\right]} = \sqrt{3}\,\tau
```

So under von Mises, a material yields in shear at τ_y = σ_y / √3 ≈ 0.577 σ_y. (Tresca predicts 0.5 σ_y.)

**Combined bending and torsion of a shaft:** at the surface, a bending stress σ and a torsional shear stress τ, so σ_vm = √(σ² + 3τ²). With σ = 120 MPa and τ = 60 MPa: σ_vm = √(14,400 + 10,800) = √25,200 ≈ **158.7 MPa**. With σ_y = 350 MPa, the yield safety factor is 350 / 158.7 ≈ **2.2**.

**Hydrostatic pressure:** σ₁ = σ₂ = σ₃ = −100 MPa gives σ_vm = 0. A metal block deep in the ocean does not yield, however high the pressure.

## Safety factor

The general idea: how far is the part from failure?

- **Yield safety factor:** SF = σ_y / σ_vm,max.
- **Ultimate safety factor:** SF = σ_ultimate / σ_max, against fracture.
- **Load factor:** the multiple of the design load at which failure occurs; the same as the stress ratio only in linear problems.
- **Reserve factor:** allowable / actual for a given failure mode (common in aerospace stress reports); **margin of safety** MS = SF − 1.

Required values depend on the industry, the confidence in loads and material data, and the consequences of failure. Civil aviation regulations, for example, require structures to support **ultimate loads**, defined as the limit (maximum expected) loads multiplied by a factor of safety of 1.5. Building and machine-design codes use their own factors, often split into separate partial factors on loads and materials.

**The "max" must be meaningful.** Exclude singular points (sharp re-entrant corners, point loads, point constraints), where the computed stress depends on the mesh rather than the physics. See [Stress concentrations and singularities](stress-concentrations-and-singularities.md).

## Predicting a safety factor with a surrogate

There are several ways to set up the prediction:

1. **Predict the stress field, then take its maximum.** Gives the location and the full picture, but the maximum is taken over a predicted field and inherits every local error.
2. **Predict the maximum stress (or the SF) directly** as a scalar output, optionally as an extra head on a field model. Simpler target, but loses the location.
3. **Classify pass/fail** against the required SF. Directly aligned with the decision, but gives no margin information.

**What goes wrong:**

- **Peaks are under-predicted.** Neural networks fit smooth, low-frequency structure first (spectral bias), and averaging losses reward getting the large, low-stress regions right. The sharp peak at a fillet, the one number the decision depends on, is systematically too low, which is *unconservative*. See [Spectral bias](spectral-bias-and-fourier-features.md#spectral-bias).
- **The location of the peak moves between designs.** The maximum of a field requires an accurate field everywhere the peak might be.
- **Labels can be wrong.** If training peaks come from singular points, or from meshes too coarse to resolve fillets, the target is mesh noise. Clean the labels: exclude zones around constraints and load points, and check mesh convergence of the peaks on a sample.
- **Nonlinearity amplifies error near the threshold.** SF = σ_y / σ_max. A 5% stress error is a 5% SF error, and if the required SF is 1.5 and the design sits at 1.52, a 5% error flips the decision.
- **Errors are heavy-tailed.** Most designs are predicted well and a few badly, and the bad ones are often the unusual geometries that matter. Average metrics hide this.
- **Material and load scatter.** σ_y itself varies between batches; the real question is the probability of failure, not a single SF.
- **Extrapolation.** Lightweighting and optimization push designs to the edge of the data, exactly where the surrogate is least reliable. See [Why optimizers exploit surrogates](surrogate-based-optimization.md#why-optimizers-exploit-surrogates).

**How to make it work:**

- **Predict the decision quantity with its uncertainty,** for example calibrated intervals on σ_max from an ensemble or [conformal prediction](conformal-prediction.md), and use the conservative end of the interval for the check.
- **Route borderline cases to the solver.** If the interval for SF crosses the required value, run FEA. Only clear passes and clear fails are decided by the surrogate. See [From uncertainty to a design decision](uncertainty-quantification.md#from-uncertainty-to-a-design-decision).
- **Weight the loss towards high-stress regions,** or train a separate head on the peak, so the model is not rewarded only for low-stress areas.
- **Evaluate at the decision level:** report the *false-pass rate* (unsafe designs called safe) separately from the false-fail rate, and the error of the peak on held-out designs, not only the field error. See [Low average error but failures where it matters](evaluation-metrics.md#low-average-error-but-failures-where-it-matters).
- **Keep sign-off with the solver.** The surrogate screens; the validated FEA certifies.

## Code: von Mises stress and a safety-factor check

```python
import numpy as np

def von_mises(s):
    """s[..., 6] = (sxx, syy, szz, sxy, syz, szx) in MPa; returns the von Mises stress."""
    sxx, syy, szz, sxy, syz, szx = np.moveaxis(s, -1, 0)
    return np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
                   + 3.0 * (sxy**2 + syz**2 + szx**2))

tau = 100.0
print(von_mises(np.array([0, 0, 0, tau, 0, 0])))               # pure shear: sqrt(3) * tau = 173.2
print(von_mises(np.array([-100, -100, -100, 0, 0, 0])))         # hydrostatic pressure: 0.0
print(von_mises(np.array([120, 0, 0, 60, 0, 0])))               # bending + torsion: 158.7

# A surrogate predicts the peak stress with an uncertainty band. Decide with the conservative end.
sigma_y, sf_required = 350.0, 1.5
for name, peak, half_width in [("design A", 180.0, 12.0), ("design B", 228.0, 15.0), ("design C", 260.0, 10.0)]:
    sf_low, sf_high = sigma_y / (peak + half_width), sigma_y / (peak - half_width)
    if sf_low >= sf_required:
        verdict = "pass (surrogate)"
    elif sf_high < sf_required:
        verdict = "fail (surrogate)"
    else:
        verdict = "borderline: run FEA"
    print(f"{name}: SF between {sf_low:.2f} and {sf_high:.2f} -> {verdict}")
```

Output:

```
173.20508075688772
0.0
158.74507866387543
design A: SF between 1.82 and 2.08 -> pass (surrogate)
design B: SF between 1.44 and 1.64 -> borderline: run FEA
design C: SF between 1.30 and 1.40 -> fail (surrogate)
```

## In an interview

**Thirty-second answer to "What is von Mises stress, and why is it used for ductile metals?"** "It is an equivalent stress that turns the six-component stress state into one number comparable with the uniaxial yield strength. It depends only on the deviatoric, shape-changing part of the stress, the distortion energy, and ignores hydrostatic pressure. That matches ductile metals, which yield by shear-driven slip that pressure alone does not cause. It is always positive and invariant to rotation, so for brittle materials, fatigue or buckling it is the wrong number."

**Likely follow-ups**

- *Von Mises in pure shear?* √3 τ, so shear yield is about 0.577 σ_y.
- *Tresca vs von Mises?* Tresca uses maximum shear and is up to about 15% more conservative.
- *How would you predict a safety factor with a surrogate?* Predict the peak with calibrated uncertainty, decide only clear passes and fails, send borderline cases to FEA, and evaluate the false-pass rate.

## References

- [MIT OpenCourseWare 2.001 Mechanics and Materials I](https://ocw.mit.edu/courses/2-001-mechanics-materials-i-fall-2006/)
- [14 CFR 25.303, Factor of safety (US airworthiness standards, Cornell LII)](https://www.law.cornell.edu/cfr/text/14/25.303)
- R. C. Juvinall and K. M. Marshek, *Fundamentals of Machine Component Design*, Wiley.
- [A Gentle Introduction to Conformal Prediction (Angelopoulos & Bates)](https://arxiv.org/abs/2107.07511)
