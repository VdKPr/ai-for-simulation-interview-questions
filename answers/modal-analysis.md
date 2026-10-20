# Modal Analysis

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Modal analysis finds the **natural frequencies** and **mode shapes** of a structure: the frequencies at which it vibrates freely, and the deformed shape of each vibration. It solves the eigenvalue problem **K φ = ω² M φ**, with stiffness matrix K and mass matrix M; each eigenvalue ω² gives a natural frequency f = ω/2π, and each eigenvector φ the corresponding mode shape. Engineers use it to avoid **resonance** (keeping natural frequencies away from the frequencies of engines, rotors, road or wind loads), as a quick check of stiffness and of model connectivity, as the basis for dynamic response calculations by **modal superposition**, and to validate models against vibration tests.

**On this page**

- [Free vibration and the eigenvalue problem](#free-vibration-and-the-eigenvalue-problem)
- [Properties of modes](#properties-of-modes)
- [What natural frequencies and mode shapes are used for](#what-natural-frequencies-and-mode-shapes-are-used-for)
- [Code: natural frequencies of a cantilever](#code-natural-frequencies-of-a-cantilever)
- [Practical points](#practical-points)
- [Why this matters for machine learning](#why-this-matters-for-machine-learning)
- [In an interview](#in-an-interview)

## Free vibration and the eigenvalue problem

**One degree of freedom first.** A mass m on a spring of stiffness k, pulled and released, oscillates at

```math
\omega = \sqrt{\frac{k}{m}}, \qquad f = \frac{\omega}{2\pi}
```

Stiffer means higher frequency; heavier means lower.

**Many degrees of freedom.** The undamped equations of motion of a discretized structure are

```math
M\,\ddot{u} + K\,u = 0 .
```

Look for solutions where every point moves in step, harmonically: u(t) = φ sin(ωt). Substituting gives

```math
\left(K - \omega^2 M\right)\varphi = 0 ,
```

a **generalized eigenvalue problem**. A non-zero φ exists only for special values of ω²: the eigenvalues. A model with n degrees of freedom has n modes; in practice only the lowest tens to hundreds are computed, with sparse eigensolvers (Lanczos, subspace iteration, AMLS for very large models).

- **Natural frequency** f_i = ω_i / 2π (in Hz): how fast mode i oscillates.
- **Mode shape** φ_i: the pattern of relative motion, for example the first bending mode of a cantilever (tip moving up and down), the second bending mode (with one node point in the middle of the span), a torsion mode, a breathing mode of a shell.
- The amplitude of a mode shape is arbitrary: modal analysis says *how* the structure likes to move, not *how much*. Amplitudes come from a forced-response analysis.

The structure is the same as in [linear buckling](buckling.md#why-linear-buckling-is-an-eigenvalue-problem), with M in place of the geometric stiffness.

## Properties of modes

- **Orthogonality.** Modes are orthogonal with respect to both matrices: φ_iᵀ M φ_j = 0 and φ_iᵀ K φ_j = 0 for i ≠ j. This is why they can be used as independent coordinates.
- **Mass normalization.** Modes are usually scaled so that φ_iᵀ M φ_i = 1; then φ_iᵀ K φ_i = ω_i².
- **Rigid-body modes.** An unsupported (free-free) structure has six modes at zero frequency in 3D: three translations and three rotations. More than six zero-frequency modes means some part is not connected.
- **Modal superposition.** Any vibration can be written as a sum of mode shapes times modal coordinates, u(t) = Σ q_i(t) φ_i. In those coordinates the equations decouple into independent single-degree-of-freedom oscillators.

## What natural frequencies and mode shapes are used for

1. **Avoiding resonance.** If a structure is excited near a natural frequency, small forces produce large vibrations, limited only by damping. Design targets therefore often say "first mode above X Hz" or "no mode within ±10% of the operating speed": a rotating machine's running speed and its harmonics, a car's engine orders, rotor blade-pass frequencies, wind or wave loading, footfall on floors.
2. **A quick stiffness check.** A low first frequency shows a weak load path or a missing support. Many specifications for brackets, mounts and spacecraft structures set a minimum first frequency precisely because it is a compact measure of stiffness relative to mass.
3. **Model checks.** Zero-frequency modes in a supposedly supported model reveal unconnected parts or missing constraints; very local modes reveal poorly modelled small features.
4. **Dynamic response by modal superposition.** Harmonic response (frequency-response functions), random vibration (response to a power spectral density, as in launch vibration or road loads), shock and transient responses are computed efficiently in modal coordinates using the lowest modes.
5. **Seismic and base-excitation design.** Modal participation factors and effective masses show which modes carry the response to ground motion; response-spectrum analysis combines them.
6. **Test correlation and model updating.** Experimental modal analysis (hammer or shaker tests) measures frequencies and mode shapes. The **modal assurance criterion** (MAC) compares test and simulated mode shapes:

```math
\text{MAC}(\varphi_a, \varphi_b) = \frac{\left|\varphi_a^{\mathsf T}\varphi_b\right|^2}{\left(\varphi_a^{\mathsf T}\varphi_a\right)\left(\varphi_b^{\mathsf T}\varphi_b\right)}
```

   MAC is 1 for identical shapes and 0 for orthogonal ones, and it is independent of sign and scale. Material properties or joint stiffnesses are then adjusted until frequencies and MAC values match.
7. **Reduced models.** Component mode synthesis (for example Craig-Bampton) represents a large component by a few modes plus its interface degrees of freedom, so assemblies can be analysed cheaply.

## Code: natural frequencies of a cantilever

Euler-Bernoulli beam elements with a consistent mass matrix, clamped at the root, compared with the exact frequencies ω_n = (β_n L)² √(EI / (ρA L⁴)), where β_n L are the roots of 1 + cos(βL) cosh(βL) = 0.

```python
import numpy as np
import scipy.linalg as sla

def cantilever_modes(n_el, EI=1.0, rhoA=1.0, L=1.0, n_modes=4):
    """Natural frequencies and mode shapes of a cantilever: K phi = omega^2 M phi."""
    le = L / n_el
    k = EI / le**3 * np.array([[12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
                               [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]])
    m = rhoA * le / 420 * np.array([[156, 22*le, 54, -13*le], [22*le, 4*le**2, 13*le, -3*le**2],
                                    [54, 13*le, 156, -22*le], [-13*le, -3*le**2, -22*le, 4*le**2]])
    ndof = 2 * (n_el + 1)
    K, M = np.zeros((ndof, ndof)), np.zeros((ndof, ndof))
    for e in range(n_el):
        d = slice(2*e, 2*e + 4)
        K[d, d] += k
        M[d, d] += m                                       # consistent mass matrix
    free = np.arange(2, ndof)                              # clamp the root: deflection and rotation
    w2, phi = sla.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    return np.sqrt(w2[:n_modes]), phi[:, :n_modes]         # omega (rad/s) and mass-normalized modes

beta_L = np.array([1.87510, 4.69409, 7.85476, 10.99554])  # roots of 1 + cos(bL) cosh(bL) = 0
exact = beta_L**2                                          # omega_n with EI = rho A = L = 1
omega, phi = cantilever_modes(20)
for i in range(4):
    print(f"mode {i+1}: FE omega = {omega[i]:8.4f}   exact = {exact[i]:8.4f}   error = {100*(omega[i]/exact[i]-1):.4f}%")
print("ratio of 2nd to 1st frequency:", round(omega[1] / omega[0], 3))
```

Output:

```
mode 1: FE omega =   3.5160   exact =   3.5160   error = 0.0004%
mode 2: FE omega =  22.0345   exact =  22.0345   error = 0.0003%
mode 3: FE omega =  61.6982   exact =  61.6973   error = 0.0016%
mode 4: FE omega = 120.9095   exact = 120.9019   error = 0.0063%
ratio of 2nd to 1st frequency: 6.267
```

Two things to notice: the FE frequencies are slightly *above* the exact ones (a displacement-based model is too stiff, and the error grows for higher modes, which need more elements per wavelength), and the second bending frequency of a cantilever is about 6.27 times the first, a ratio worth remembering.

## Practical points

- **Mass matters as much as stiffness.** Missing masses (fluids, batteries, attached equipment modelled as point masses with rigid links) shift frequencies. Check the total mass of the model.
- **Boundary conditions dominate.** A bracket's first frequency depends strongly on how stiff its mounting really is; "fixed" supports over-predict frequencies.
- **Prestress changes frequencies.** Tension stiffens (a guitar string, a spinning turbine blade under centrifugal load), compression softens; prestressed modal analysis includes the geometric stiffness.
- **Damping is separate.** Undamped modal analysis gives frequencies and shapes; damping (often modelled as a few percent of critical per mode, or as Rayleigh damping) enters the forced-response calculation.
- **Close and repeated frequencies** occur in symmetric structures (a round plate, a square tube); their modes can rotate into any combination, so comparing individual mode shapes between models needs care.

## Why this matters for machine learning

- **Natural frequencies are excellent scalar targets:** smooth in the design parameters (except where modes cross), cheap to compute, and design-relevant. Non-dimensionalize them with √(E/ρ)/L so the model does not have to learn the material scaling. See [Why non-dimensionalize](nondimensionalization-and-normalization.md#why-non-dimensionalize).
- **Mode shapes need sign- and scale-invariant losses.** φ and −φ are the same mode, and amplitudes are arbitrary. Use a MAC-based loss, or align signs before comparing. Repeated frequencies need comparison of subspaces, not individual vectors.
- **Mode ordering can swap.** As a design changes, two modes can cross or veer: "mode 3" in one design may correspond to "mode 4" in another. Match modes by MAC before training on "the third frequency", or predict the sorted set of frequencies with a permutation-invariant loss.
- **Modes are a natural reduced basis.** Like POD modes, structural modes give a compact basis for dynamic responses, so a surrogate can predict modal coordinates instead of full fields. See [Reduced-order models](reduced-order-models.md).

## In an interview

**Thirty-second answer.** "Modal analysis solves K φ = ω² M φ for the free, undamped vibrations of a structure. The eigenvalues give the natural frequencies and the eigenvectors the mode shapes, with arbitrary amplitude. We use it to keep natural frequencies away from excitation frequencies to avoid resonance, as a stiffness and connectivity check (rigid-body modes reveal unconnected parts), as the basis for modal superposition in harmonic and random vibration, and to correlate models with tests using frequencies and the MAC."

**Likely follow-ups**

- *How many zero-frequency modes does a free-free 3D body have?* Six: three translations and three rotations.
- *A bracket must have its first mode above 200 Hz but is at 150 Hz. What do you do?* Increase stiffness where the first mode shape shows the most deformation (ribs, thicker walls, shorter spans), or reduce mass where the mode shape moves most; check the mounting stiffness assumption.
- *How would you compare predicted and simulated mode shapes?* With the MAC, which ignores sign and scale.

## References

- R. D. Cook, D. S. Malkus, M. E. Plesha and R. J. Witt, *Concepts and Applications of Finite Element Analysis*, 4th edition, Wiley, 2002 (consistent mass matrices).
- D. J. Ewins, *Modal Testing: Theory, Practice and Application*, 2nd edition, Research Studies Press, 2000.
- R. R. Craig Jr. and A. J. Kurdila, *Fundamentals of Structural Dynamics*, 2nd edition, Wiley, 2006.
