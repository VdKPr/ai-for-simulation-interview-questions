# Reduced-Order Models

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A reduced-order model (ROM) replaces a simulation with millions of unknowns by one with a handful, by finding a **low-dimensional basis** in which the solutions live. **Proper orthogonal decomposition (POD)** finds the optimal linear basis from **snapshots**: stack solution fields as columns of a matrix, subtract the mean, and take the left singular vectors of the SVD; the singular values say how much energy each mode carries. An **intrusive** ROM projects the governing equations onto the POD modes (Galerkin projection) and solves the small system, which needs access to the solver's operators. A **non-intrusive** ROM treats the solver as a black box and learns a map from parameters to POD coefficients (interpolation, GPs, neural networks). **Dynamic mode decomposition (DMD)** fits the best linear time-stepping operator to snapshot pairs and returns modes that each oscillate and grow or decay at a single frequency. Linear subspaces **fail for transport-dominated problems**: a moving front, shock or vortex needs many modes because each position is a new direction in the space (slowly decaying Kolmogorov n-width). Nonlinear manifolds learned by **autoencoders**, or shifted and transported bases, address this.

**On this page**

- [Proper orthogonal decomposition](#proper-orthogonal-decomposition)
- [Intrusive vs non-intrusive ROMs](#intrusive-vs-non-intrusive-roms)
- [Dynamic mode decomposition](#dynamic-mode-decomposition)
- [When a linear subspace fails](#when-a-linear-subspace-fails)
- [ROM or deep surrogate?](#rom-or-deep-surrogate)
- [Code: POD with the SVD](#code-pod-with-the-svd)
- [In an interview](#in-an-interview)

## Proper orthogonal decomposition

**Setup.** Collect n_s snapshots, solutions for different parameters or at different times, each a vector of n degrees of freedom (n can be 10⁶). Arrange them as the columns of the snapshot matrix S ∈ ℝ^{n × n_s} and subtract the mean snapshot ū.

**Compute the SVD.**

```math
S - \bar u\,\mathbf 1^\top = U \Sigma V^\top
```

- The columns of U are the **POD modes** φ_k: orthonormal spatial patterns, ordered by importance.
- The singular values σ_k measure importance: σ_k² is the "energy" (variance) captured by mode k.
- Keep the first r modes such that Σ_{k≤r} σ_k² / Σ_k σ_k² exceeds a threshold, for example 99.9%.

**Why it is optimal.** Among all r-dimensional linear subspaces, the span of the first r POD modes minimizes the total squared projection error of the snapshots (Eckart-Young theorem). The reconstruction error is exactly Σ_{k>r} σ_k².

**Practical points.**

- **Method of snapshots:** when n ≫ n_s, compute the small n_s × n_s correlation matrix SᵀS, its eigenvectors give V, and U = SVΣ⁻¹. The thin SVD (`full_matrices=False`) does the equivalent.
- **Weighting:** on non-uniform meshes, use the mass matrix (cell volumes) as the inner product, otherwise refined regions dominate the energy. Equivalently, scale rows by √(cell volume) before the SVD.
- **Scaling multiple fields:** normalize velocity, pressure and temperature before stacking them, or the field with the largest numbers dominates.
- **Large data:** randomized SVD, incremental SVD, or distributed tall-skinny QR for snapshot matrices that do not fit in memory.
- **Cost of a solution in the basis:** u ≈ ū + Φ a, with r coefficients a = Φᵀ(u − ū).

## Intrusive vs non-intrusive ROMs

**Intrusive (Galerkin projection).** Substitute u ≈ ū + Φa into the discretized equations and project them onto the modes.

- For a linear system A(μ)u = f: solve (ΦᵀA(μ)Φ) a = Φᵀf, an r × r system instead of n × n.
- For time-dependent problems du/dt = F(u): da/dt = Φᵀ F(ū + Φa).
- **Advantages:** the physics (the operator) is still in the model, so it extrapolates better than pure regression and can be accurate with few snapshots; error estimators and certified reduced-basis methods exist.
- **Difficulties:**
  - needs access to the solver's matrices or residual (impossible with many commercial codes);
  - for nonlinear problems, evaluating ΦᵀF(ū + Φa) still costs O(n) per step unless **hyper-reduction** (DEIM, gappy POD, ECSW) samples a few points;
  - Galerkin ROMs of convection-dominated flows can be unstable (energy pile-up in truncated modes), needing stabilization or closure models.

**Non-intrusive (data-driven).** Run the solver for training parameters, compute POD coefficients, and **learn** the map μ ↦ a(μ) with interpolation (splines, RBFs), Gaussian processes or neural networks. For time-dependent problems, learn a ↦ a_next (or da/dt) with a recurrent model, DMD or SINDy. See [SINDy and Koopman](sindy-and-koopman.md).

- **Advantages:** works with any solver, including commercial ones; simple; GPs give uncertainty on each coefficient.
- **Difficulties:** no physics in the reduced dynamics, so it interpolates but extrapolates poorly; needs enough snapshots to learn the coefficient map; errors in coefficients of high modes can produce non-physical fields.

| | Intrusive (Galerkin) | Non-intrusive (POD + regression) |
| --- | --- | --- |
| Needs solver internals | Yes (operators, residuals) | No |
| Physics in the online model | Yes | No |
| Nonlinear cost | Needs hyper-reduction | Cheap (just regression) |
| Typical tools | Reduced-basis methods, libROM, pyMOR | Any regression library; Ansys optiSLang-style workflows |
| Typical risk | Instability for convection | Poor extrapolation |

## Dynamic mode decomposition

**Idea.** Given snapshots x₀, x₁, …, x_m of a system sampled at a fixed time step, find the linear operator A that best maps each snapshot to the next, X₂ ≈ A X₁, with X₁ = [x₀ … x_{m−1}] and X₂ = [x₁ … x_m]. Its eigenvectors are the **DMD modes**; its eigenvalues λ say how each mode evolves: |λ| is growth or decay per step, arg λ / Δt is the frequency.

**Exact DMD (Schmid 2010; Tu et al. 2014).**

1. Thin SVD of X₁ = UΣVᵀ, truncated to rank r.
2. Project the operator onto the POD modes: Ã = Uᵀ X₂ V Σ⁻¹ (r × r).
3. Eigen-decompose Ã W = W Λ; the DMD eigenvalues are Λ.
4. DMD modes: Φ = X₂ V Σ⁻¹ W.
5. Forecast: x_k ≈ Φ Λᵏ b, with amplitudes b from the first snapshot.

**POD vs DMD.** POD modes are ordered by energy and each contains a mix of frequencies; DMD modes each have a single frequency and growth rate, so they separate, for example, a vortex-shedding mode from its harmonics. DMD is a finite-dimensional approximation of the **Koopman operator**. See [The Koopman operator](sindy-and-koopman.md#the-koopman-operator).

**Uses and limits.** Identify dominant frequencies and coherent structures in CFD and experiments (PIV), forecast near-periodic flows, and build linear control models (DMD with control). It struggles with strongly nonlinear transients, travelling waves (same problem as POD), and noise (which biases eigenvalues; variants such as optimized DMD and total-least-squares DMD help).

## When a linear subspace fails

**The diagnosis: Kolmogorov n-width.** The best possible error of any n-dimensional linear subspace for a set of solutions decays fast for diffusion-dominated problems (smooth dependence on parameters) and slowly for transport-dominated ones. A moving shock, a travelling wave, a vortex shed at different positions or a moving contact front: each shifted copy is nearly orthogonal to the others, so the singular values decay slowly and many modes are needed. The [code below](#code-pod-with-the-svd) shows a diffusion family needing 3 modes and a moving pulse needing 10 to 86, depending on how sharp it is.

**Symptoms.** Slowly decaying singular values; ROM reconstructions with oscillations (Gibbs-like ringing) around fronts; "ghost" copies of a feature at several positions.

**Remedies.**

1. **Autoencoder ROMs:** a neural network encoder maps the field to a few latent variables and a decoder maps back, so solutions lie on a curved **nonlinear manifold** rather than a flat subspace. Lee and Carlberg (2020) projected the dynamics onto such a manifold (a nonlinear analogue of Galerkin projection) and beat POD-Galerkin on advection-dominated problems with very few latent variables. Latent dynamics can then be learned (recurrent networks, neural ODEs, SINDy in the latent space).
2. **Transported or shifted bases:** move the coordinate frame with the feature (shifted POD, method of freezing, registration), so the solution looks steady in the moving frame.
3. **Local bases:** different POD bases for different regions of parameter space or time windows.
4. **Neural operators and field surrogates** that do not assume a linear basis. See [Neural operators](neural-operators.md).

## ROM or deep surrogate?

| Situation | Better choice |
| --- | --- |
| Parametric study, smooth dependence, solver access, few snapshots | Intrusive POD-Galerkin or reduced-basis method |
| Parametric study with a commercial black-box solver | POD + GP (non-intrusive), with uncertainty |
| Many varied geometries (different meshes) | Deep surrogate (GNN, transformer); POD needs a common mesh |
| Transport-dominated dynamics | Autoencoder ROM or neural operator |
| Need certified error bounds | Reduced-basis methods with a posteriori estimators |
| Real-time digital twin of one asset | ROM (fast, stable, small), often with sensor data assimilation |

POD needs all snapshots on **the same mesh**, which is why ROMs are popular for parametric studies and digital twins of one product, and deep geometric surrogates for many different shapes.

## Code: POD with the SVD

A full-order model, steady 1D diffusion with a parameter-dependent conductivity bump, solved with finite differences on 400 nodes. Twelve snapshots give the POD basis; an intrusive Galerkin ROM and a non-intrusive coefficient interpolation then predict a new parameter. A family of moving pulses shows the linear-subspace failure, and DMD recovers the eigenvalues of a linear system from snapshots.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

def pod(snapshots, energy=0.999):
    """snapshots: (n_dof, n_snapshots). Returns the mean, the first r modes and the snapshot coefficients,
    with r the smallest number of modes that captures the requested fraction of the fluctuation energy."""
    mean = snapshots.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(snapshots - mean, full_matrices=False)
    r = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), energy) + 1)
    modes = U[:, :r]
    coeffs = modes.T @ (snapshots - mean)                    # project onto the modes
    return mean, modes, coeffs                               # reconstruct: mean + modes @ coeffs

# Full-order model: steady 1D diffusion -(k(x; mu) u')' = 1 on (0, 1), u(0) = u(1) = 0, finite differences.
n = 400
x = np.linspace(0, 1, n + 2)[1:-1]
h = x[1] - x[0]
def full_model_matrix(mu):
    k_face = 1 + mu * np.exp(-((np.linspace(0, 1, n + 1) - 0.3) / 0.1) ** 2)   # conductivity with a bump of size mu
    main = (k_face[:-1] + k_face[1:]) / h ** 2
    return sp.diags([-k_face[1:-1] / h ** 2, main, -k_face[1:-1] / h ** 2], [-1, 0, 1], format="csc")

f = np.ones(n)
train_mu = np.linspace(0, 10, 12)
S = np.stack([spla.spsolve(full_model_matrix(m), f) for m in train_mu], axis=1)     # 12 snapshots
mean, modes, coeffs = pod(S, energy=0.99999)
print(f"diffusion family: {modes.shape[1]} POD modes keep 99.999% of the energy of {S.shape[1]} snapshots")

test_mu = 6.3
u_true = spla.spsolve(full_model_matrix(test_mu), f)
Phi = np.hstack([mean, modes])                               # basis: the mean plus the POD modes
A = full_model_matrix(test_mu)
u_galerkin = Phi @ np.linalg.solve(Phi.T @ (A @ Phi), Phi.T @ f)            # intrusive: project the operator
u_interp = mean[:, 0] + modes @ np.array([CubicSpline(train_mu, c)(test_mu) for c in coeffs])   # non-intrusive
rel = lambda u: np.linalg.norm(u - u_true) / np.linalg.norm(u_true)
print(f"mu = {test_mu}: Galerkin ROM ({Phi.shape[1]} unknowns instead of {n}) error {rel(u_galerkin):.2e}; "
      f"coefficient interpolation error {rel(u_interp):.2e}")

# A transport-dominated family: a pulse whose position changes with the parameter.
xs = np.linspace(0, 1, 400)
for width in [0.1, 0.03, 0.01]:
    pulses = np.stack([np.exp(-((xs - c) / width) ** 2) for c in np.linspace(0.2, 0.8, 100)], axis=1)
    print(f"moving pulse, width {width}: {pod(pulses, energy=0.99999)[1].shape[1]} modes for 99.999% of the energy")

# Dynamic mode decomposition: recover the eigenvalues of a linear system from snapshots of a 400-dof field.
rng = np.random.default_rng(0)
lam = np.array([0.95 * np.exp(1j * 0.3), 0.95 * np.exp(-1j * 0.3), 0.80, 0.60])     # true discrete-time eigenvalues
basis = np.linalg.qr(rng.normal(size=(400, 4)))[0]
spatial = basis @ np.array([[1, 1j, 0, 0], [1, -1j, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
X = np.real(np.stack([spatial @ (lam ** k * np.array([1, 1, 0.5, 0.5])) for k in range(30)], axis=1))
X1, X2 = X[:, :-1], X[:, 1:]
U, s, Vh = np.linalg.svd(X1, full_matrices=False)
r = 4
A_tilde = U[:, :r].conj().T @ X2 @ Vh[:r].conj().T / s[:r]          # the step operator projected onto r POD modes
dmd_eigs = np.linalg.eigvals(A_tilde)
print("DMD eigenvalues: ", np.round(sorted(dmd_eigs, key=lambda z: (-abs(z), z.imag)), 4))
print("true eigenvalues:", np.round(sorted(lam, key=lambda z: (-abs(z), z.imag)), 4))
```

Output:

```
diffusion family: 3 POD modes keep 99.999% of the energy of 12 snapshots
mu = 6.3: Galerkin ROM (4 unknowns instead of 400) error 4.84e-05; coefficient interpolation error 3.51e-05
moving pulse, width 0.1: 10 modes for 99.999% of the energy
moving pulse, width 0.03: 30 modes for 99.999% of the energy
moving pulse, width 0.01: 86 modes for 99.999% of the energy
DMD eigenvalues:  [0.9076-0.2807j 0.9076+0.2807j 0.8   +0.j     0.6   +0.j    ]
true eigenvalues: [0.9076-0.2807j 0.9076+0.2807j 0.8   +0.j     0.6   +0.j    ]
```

**What it shows.**

- The diffusion family is very low-dimensional: 3 modes capture 99.999% of the variation, and both ROMs predict an unseen parameter with errors around 10⁻⁵ using 4 unknowns instead of 400.
- The intrusive ROM solved the projected equations; the non-intrusive one only interpolated coefficients. On this smooth, interpolative problem both work; the difference appears when extrapolating or when snapshots are scarce.
- A moving pulse needs more modes the sharper it is, and the count would keep growing as the width shrinks: the Kolmogorov n-width problem in miniature.
- DMD recovers the four eigenvalues (an oscillating, slowly decaying pair and two decaying modes) to four decimals from 30 snapshots of a 400-dimensional field.

## In an interview

**Thirty-second answer to "What is POD, and how do you compute it from snapshots?"** "Stack solution fields from different parameters or times as columns of a matrix, subtract the mean, and take the thin SVD. The left singular vectors are the POD modes, an orthonormal basis ordered by energy, and the squared singular values say how much variance each captures; you keep enough modes for, say, 99.9%. It is the optimal linear basis for those snapshots. Then either project the equations onto the modes, an intrusive Galerkin ROM, or regress the mode coefficients on the parameters, a non-intrusive ROM. It needs a common mesh, and it struggles with moving fronts, where the singular values decay slowly."

**Likely follow-ups**

- *Intrusive vs non-intrusive?* Intrusive projects the solver's operators and keeps physics but needs solver access and hyper-reduction for nonlinear terms; non-intrusive learns the coefficient map from data with any solver.
- *What is DMD?* The best-fit linear step operator between consecutive snapshots, projected onto POD modes; each DMD mode has one frequency and growth rate.
- *When does a linear subspace fail?* Transport-dominated problems; use autoencoders, shifted bases or local bases.

## References

- S. L. Brunton and J. N. Kutz, *Data-Driven Science and Engineering*, Cambridge University Press (chapters on SVD, POD, DMD and ROMs; [book website with videos](https://databookuw.com/)).
- P. J. Schmid, "Dynamic mode decomposition of numerical and experimental data", *Journal of Fluid Mechanics*, 2010.
- [On Dynamic Mode Decomposition: Theory and Applications (Tu et al.)](https://arxiv.org/abs/1312.0041)
- [Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders (Lee & Carlberg)](https://arxiv.org/abs/1812.08373)
- J. S. Hesthaven, G. Rozza and B. Stamm, *Certified Reduced Basis Methods for Parametrized Partial Differential Equations*, Springer, 2016.
- [PyDMD](https://github.com/PyDMD/PyDMD), [pyMOR](https://pymor.org/) and [libROM](https://www.librom.net/)
