# Fatigue

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Fatigue is failure under **repeated** loading at stresses well below the static strength. Each load cycle can grow a microscopic crack a little, usually starting at a stress concentration such as a fillet, a hole, a weld toe or a surface scratch, until the remaining section fractures. Fatigue life is estimated from **S-N curves** (stress amplitude against cycles to failure), corrected for mean stress, and accumulated over variable loading with **rainflow counting** and **Miner's rule**. It is hard to predict with a surrogate for four reasons: life depends on the *local peak* stress raised to a high power, so a small stress error becomes a large life error (under-predicting stress by 10% can over-predict life by 1.4 to 2.9 times, depending on the S-N slope); it depends on details the training data rarely resolve (surface finish, weld geometry, residual stress); it depends on the load history, not one load case; and real fatigue lives scatter widely even for identical parts. A safer design is to let the surrogate predict stresses and apply a validated fatigue method on top.

**On this page**

- [What fatigue is](#what-fatigue-is)
- [Stress-life: S-N curves](#stress-life-s-n-curves)
- [Mean stress](#mean-stress)
- [Variable loading: rainflow counting and Miner's rule](#variable-loading-rainflow-counting-and-miners-rule)
- [What else changes fatigue life](#what-else-changes-fatigue-life)
- [Why fatigue life is hard for a surrogate](#why-fatigue-life-is-hard-for-a-surrogate)
- [Code: Miner's rule and the sensitivity of life to stress errors](#code-miners-rule-and-the-sensitivity-of-life-to-stress-errors)
- [How to use ML for fatigue anyway](#how-to-use-ml-for-fatigue-anyway)
- [In an interview](#in-an-interview)

## What fatigue is

A paper clip bent back and forth breaks after a few dozen bends, although no single bend breaks it. Metal structures under vibration, rotation, pressure cycles or traffic do the same over thousands to billions of cycles at much lower stresses.

**Three stages:**

1. **Crack initiation** at a local stress raiser: a notch, fillet, hole, weld toe, inclusion or surface flaw. Cyclic slip in the most stressed grains forms a microcrack.
2. **Crack propagation:** the crack grows a little each cycle, perpendicular to the largest principal stress, often leaving visible "beach marks" on the fracture surface.
3. **Final fracture:** the remaining cross-section can no longer carry the load and fails suddenly.

Fatigue is behind many well-known failures, from 19th-century railway axles (which led to Wöhler's tests and the first S-N curves) to aircraft fuselages under pressurization cycles. It is the dominant failure mode of rotating machinery, vehicles, welded structures, offshore structures and bolts.

## Stress-life: S-N curves

An **S-N curve** (Wöhler curve) plots the stress amplitude S_a against the number of cycles to failure N, on log-log axes, from constant-amplitude tests. Over much of its range it is close to a straight line, the **Basquin relation**:

```math
N = C\,S_a^{-m} \qquad \text{(equivalently } S_a = \sigma_f' (2N)^b, \; m = -1/b \text{)}
```

- **m is large.** Welded steel details have S-N slopes of about m = 3 in design standards; smooth, polished specimens have much flatter curves with m of 10 or more. A large m means life is very sensitive to stress.
- **Endurance limit.** Many steels show a stress amplitude below which they can survive very large numbers of cycles (a "knee" in the curve, often roughly half the ultimate tensile strength for smooth specimens of ordinary steels). Aluminium alloys show no clear endurance limit, so their design uses a life such as 10⁷ or 10⁸ cycles.
- **High-cycle vs low-cycle fatigue.** Above about 10⁴ to 10⁵ cycles, stresses stay mostly elastic and the stress-life approach works. At fewer cycles, local plasticity dominates and the **strain-life** approach (Coffin-Manson) is used, with local elastic-plastic strains.

Design curves are usually set two or three standard deviations below the mean test data, because fatigue lives scatter widely: identical specimens at the same stress can differ in life by a factor of several.

## Mean stress

A cycle is defined by its amplitude S_a and its mean S_m (or by the stress ratio R = S_min / S_max). A **tensile** mean stress shortens life; a compressive mean stress lengthens it. Classic corrections convert a cycle with a mean stress into an equivalent fully reversed amplitude:

- **Goodman:** S_a / S_e + S_m / S_u = 1 (linear; widely used, somewhat conservative);
- **Gerber:** a parabola in S_m / S_u (closer to ductile-metal data);
- **Soderberg:** uses the yield strength instead of the ultimate strength (most conservative).

This is why compressive residual stresses from shot peening or rolling improve fatigue life, and tensile residual stresses from welding harm it.

## Variable loading: rainflow counting and Miner's rule

Real load histories are irregular: a car on a rough road, a wind turbine in gusty wind. Two steps turn them into a damage estimate:

1. **Rainflow counting** decomposes the stress history into a set of closed cycles, each with an amplitude and a mean. It is the standard algorithm (standardized in ASTM E1049) because it pairs up the reversals in a way that matches how materials respond to cycles.
2. **The Palmgren-Miner linear damage rule** sums the fraction of life used by each group of cycles:

```math
D = \sum_i \frac{n_i}{N_i}
```

   where n_i is the number of counted cycles at stress level i and N_i the life at that level from the S-N curve. Failure is predicted when D reaches 1.

Miner's rule ignores load-sequence effects (a large overload followed by small cycles behaves differently from the reverse), and measured damage sums at failure scatter well away from 1, which is why some standards use a lower allowable damage. It remains the industry workhorse.

**Multiaxial fatigue** (stresses that change direction during the cycle, as in crankshafts and suspension parts) needs critical-plane methods, which search for the plane that accumulates the most damage.

## What else changes fatigue life

- **Notches:** the fatigue notch factor K_f is usually smaller than the elastic K_t (materials are less sensitive to very sharp notches than the elastic peak suggests), described by a notch sensitivity q. See [Stress concentrations](stress-concentrations-and-singularities.md#stress-concentration-factors).
- **Surface finish:** machined, forged or as-cast surfaces lower the endurance limit compared with polished specimens.
- **Size:** larger parts have more chance of a critical flaw.
- **Residual stresses** from welding, forming, shot peening or heat treatment.
- **Environment:** corrosion can remove the endurance limit entirely; temperature changes properties.
- **Welds:** design codes classify weld details into categories with their own S-N curves, and evaluate stresses with defined methods (nominal, hot-spot or effective notch stress) precisely because the stress at a weld toe in an FE model is mesh-dependent.

## Why fatigue life is hard for a surrogate

1. **Life is extremely sensitive to the local stress.** With N ∝ S^(−m), a stress error δ becomes a life factor of about (1 − δ)^(−m). Under-predicting the stress by 10% over-predicts life by 1.37× for m = 3 and 2.87× for m = 10 (see the code below). Smooth neural networks *under*-predict sharp peaks, which is the unsafe direction. See [Spectral bias](spectral-bias-and-fourier-features.md#spectral-bias).
2. **The right stress is a local detail.** The crack starts at a fillet radius, a weld toe or a surface flaw. Training meshes that do not resolve those details produce peak stresses that are under-resolved or singular, so the labels are themselves uncertain. See [Why the peak stress rises with mesh refinement](stress-concentrations-and-singularities.md#why-the-peak-stress-rises-with-mesh-refinement).
3. **Fatigue depends on the whole load history.** Amplitudes, means, sequences and multiaxiality all matter. A surrogate trained on one static load case knows none of that.
4. **Life spans orders of magnitude and scatters widely.** The target should be log-life, and the inherent scatter (a factor of several between identical parts) means "ground truth" from tests is noisy. Data from physical fatigue tests is scarce and expensive.
5. **Many factors are not in the geometry.** Surface finish, residual stress, material batch and environment change life but rarely appear as inputs.
6. **Extrapolation to long lives.** The design point (say 10⁸ cycles) is often beyond the test data, where the S-N curve itself is extrapolated.

## Code: Miner's rule and the sensitivity of life to stress errors

```python
def cycles_to_failure(stress_amplitude, C=1e12, m=3.0):
    """Basquin-type S-N curve N = C * S^-m (S in MPa). m = 3 is typical of welded steel details."""
    return C * stress_amplitude ** (-m)

def miner_damage(spectrum, scale=1.0, m=3.0):
    """Palmgren-Miner rule: damage D = sum(n_i / N_i) over the blocks of a load spectrum."""
    return sum(n / cycles_to_failure(scale * s, m=m) for s, n in spectrum)

spectrum = [(80.0, 1e5), (150.0, 1e4), (250.0, 1e2)]          # (stress amplitude in MPa, cycles) per block
D = miner_damage(spectrum)
print(f"damage per block = {D:.4f} -> predicted life = {1 / D:.1f} blocks")

print("stress under-predicted by | m = 3 | m = 5 | m = 10   (factor by which life is over-predicted)")
for error in [0.02, 0.05, 0.10, 0.20]:
    factors = [miner_damage(spectrum, 1.0, m) / miner_damage(spectrum, 1 - error, m) for m in (3.0, 5.0, 10.0)]
    print(f"{error:>24.0%} | " + " | ".join(f"{f:5.2f}" for f in factors))
```

Output:

```
damage per block = 0.0865 -> predicted life = 11.6 blocks
stress under-predicted by | m = 3 | m = 5 | m = 10   (factor by which life is over-predicted)
                      2% |  1.06 |  1.11 |  1.22
                      5% |  1.17 |  1.29 |  1.67
                     10% |  1.37 |  1.69 |  2.87
                     20% |  1.95 |  3.05 |  9.31
```

A stress error of a few percent, which would be excellent for a field surrogate, becomes a life error of tens of percent. A 20% under-prediction on a smooth-specimen curve (m = 10) over-predicts life by more than nine times.

## How to use ML for fatigue anyway

- **Predict stresses, not lives.** Let the surrogate predict the stress field or the hot-spot stresses for unit load cases, then apply the validated fatigue chain (load history, rainflow, mean-stress correction, S-N curve, Miner) as a deterministic post-processing step. This keeps the well-understood physics explicit and lets fatigue engineers audit the method.
- **Exploit linearity.** For linear elastic structures, stresses for a load history are a superposition of unit-load stress fields scaled by the load time histories. A surrogate needs to learn only the unit cases.
- **Give the peaks special treatment.** A dedicated head and loss weighting for hot-spot stresses, calibrated uncertainty on them, and evaluation at the critical locations, not on field averages.
- **Be conservative at the decision.** Use the upper end of the predicted stress interval, and send designs whose damage is close to the limit to full FEA and fatigue assessment. See [From uncertainty to a design decision](uncertainty-quantification.md#from-uncertainty-to-a-design-decision).
- **Use ML where data is rich:** for example, predicting load spectra from operational data, or hot-spot locations to guide meshing, rather than replacing the fatigue calculation.

## In an interview

**Thirty-second answer to "What is fatigue, and why is fatigue life hard to predict with a surrogate?"** "Fatigue is crack initiation and growth under repeated loading below the static strength, usually starting at a stress concentration. Life comes from S-N curves with mean-stress correction, and for variable loads from rainflow counting and Miner's rule. It is hard for a surrogate because life goes like stress to a power of about 3 to 10, so a 10% under-predicted peak can mean 40% to almost three times too much life; the peak is a local detail that neural networks smooth and that coarse training meshes do not resolve; and life depends on the load history, surface, welds and residual stress, with large intrinsic scatter. I would have the surrogate predict stresses and keep the fatigue calculation explicit."

**Likely follow-ups**

- *What is Miner's rule, and its main weakness?* Linear damage summation n/N to D = 1; it ignores load sequence effects and has large scatter at failure.
- *Why do welded joints use hot-spot stresses?* The stress at a weld toe in an FE model is mesh-dependent (geometrically singular); hot-spot methods define a reproducible extrapolation.
- *Does a compressive mean stress help?* Yes: it delays crack growth, which is why shot peening improves fatigue life.

## References

- J. A. Bannantine, J. J. Comer and J. L. Handrock, *Fundamentals of Metal Fatigue Analysis*, Prentice Hall, 1990.
- R. I. Stephens, A. Fatemi, R. R. Stephens and H. O. Fuchs, *Metal Fatigue in Engineering*, 2nd edition, Wiley, 2000.
- ASTM E1049, *Standard Practices for Cycle Counting in Fatigue Analysis*.
- [rainflow: an open-source Python package for rainflow cycle counting](https://github.com/iamlikeme/rainflow)
