# SINDy and the Koopman Operator

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Both methods learn **dynamics** from time-series data rather than a static input-output map. **SINDy** (Sparse Identification of Nonlinear Dynamics; Brunton, Proctor and Kutz, PNAS 2016) assumes the governing equation dx/dt = f(x) has **only a few terms**, builds a large library of candidate functions (constants, polynomials, trigonometric terms), and uses **sparse regression** to keep the few that explain the measured derivatives. The result is an interpretable equation, for example the Lorenz system recovered term by term from noisy data. The **Koopman operator** takes the opposite route: instead of a nonlinear equation in the state, it describes the evolution of **observables** (functions of the state), and that evolution is always **linear**, at the cost of being infinite-dimensional. Finding a few observables in which the dynamics become (approximately) linear gives linear prediction, spectral analysis and control for a nonlinear system. DMD and extended DMD approximate it with chosen observables; **deep Koopman** autoencoders (Lusch, Kutz and Brunton, 2018) learn the observables.

**On this page**

- [SINDy](#sindy)
- [The Koopman operator](#the-koopman-operator)
- [SINDy, Koopman or a neural surrogate?](#sindy-koopman-or-a-neural-surrogate)
- [Code: recovering the Lorenz equations and a Koopman embedding](#code-recovering-the-lorenz-equations-and-a-koopman-embedding)
- [In an interview](#in-an-interview)

## SINDy

**The setup.** Measure the state x(t) ∈ ℝⁿ at many times and estimate the derivatives dx/dt. Build a library matrix Θ(X) whose columns are candidate functions evaluated on the data: 1, x, y, z, x², xy, …, sin x, …. Then solve

```math
\dot X \approx \Theta(X)\,\Xi, \qquad \Xi \text{ sparse}
```

Each column of Ξ gives the coefficients of one equation; sparsity means most are exactly zero.

**The algorithm (sequentially thresholded least squares, STLSQ).**

1. Solve the least-squares problem for Ξ.
2. Set every coefficient smaller than a threshold λ to zero.
3. Re-solve least squares using only the surviving terms.
4. Repeat until the set of terms stops changing.

The threshold controls the trade-off between simplicity and accuracy, chosen by cross-validation or an information criterion. Alternatives include LASSO, sparse relaxed regularized regression (SR3) and ensemble SINDy (bootstrapping for robustness and uncertainty).

**What the paper demonstrated.** Linear and nonlinear oscillators, the chaotic Lorenz system, and vortex shedding behind a cylinder, where SINDy recovered a reduced model of the shedding dynamics (in POD coordinates) that took the fluid-mechanics community nearly 30 years to establish.

**Practical issues.**

- **Derivatives from noisy data** are the main weakness: finite differences amplify noise. Use smoothing (total-variation regularized differentiation, Savitzky-Golay filters), the weak (integral) formulation of SINDy, or measured derivatives.
- **The right coordinates matter.** SINDy finds sparse equations only if they exist in the chosen variables. For PDE fields, first reduce with POD or an autoencoder (SINDy autoencoders learn coordinates and equations together).
- **The library must contain the right terms.** Missing terms cannot be discovered; too many correlated terms make the regression ill-conditioned.
- **Extensions:** SINDy with control inputs (SINDYc), implicit SINDy for rational functions, PDE-FIND for partial differential equations (candidate terms like u_xx and u·u_x), and constraints such as energy conservation.

**When it is useful in engineering.** Discovering or checking reduced models of a dominant mechanism (a flutter mode, shedding, a chemical reaction network), building interpretable models from test data, and identifying closure terms. It is not a field surrogate for a 3D CFD simulation.

## The Koopman operator

**Definition.** For a discrete-time system x_{k+1} = F(x_k), the Koopman operator 𝒦 acts on observables g (scalar functions of the state) by composition:

```math
(\mathcal K g)(x) = g\big(F(x)\big)
```

It advances any measurement one step in time. Because (𝒦(αg + βh)) = α𝒦g + β𝒦h, it is **linear**, even when F is strongly nonlinear. The price: it acts on a space of functions, so it is generally infinite-dimensional.

**Why it is useful.** If we find a finite set of observables z = (g₁(x), …, g_m(x)) that the operator maps into their own span (a **Koopman-invariant subspace**), then z_{k+1} = K z_k with a finite matrix K, and nonlinear dynamics become linear in the new coordinates. Then:

- prediction is matrix powers, z_k = K^k z₀;
- the eigenvalues of K give frequencies and growth rates; its **eigenfunctions** are intrinsic coordinates that each evolve as a single exponential;
- linear control and estimation (LQR, Kalman filters) apply.

**Approximations.**

| Method | Observables | Notes |
| --- | --- | --- |
| DMD | The state itself (linear observables) | Captures linear dynamics; fails for genuinely nonlinear systems. See [Dynamic mode decomposition](reduced-order-models.md#dynamic-mode-decomposition) |
| Extended DMD (EDMD) | A chosen dictionary: polynomials, radial basis functions, … | Least-squares fit of K in the dictionary; the dictionary choice is everything |
| Kernel DMD | Implicit features through a kernel | Large dictionaries without explicit features |
| Deep Koopman autoencoders | Learned by a network | Encoder φ(x) = z, linear K, decoder back to x; loss on reconstruction, linear prediction in z and prediction in x |

**Deep learning approach (Lusch, Kutz and Brunton, Nature Communications 2018).** A modified autoencoder learns coordinates in which the dynamics are globally linear, at the intrinsic rank of the system, and is interpretable by construction (its latent variables are Koopman eigenfunctions). For systems with a **continuous spectrum** (the simple pendulum, whose frequency depends on amplitude), a fixed linear matrix cannot work; they added an **auxiliary network** that outputs the eigenvalues as a function of the latent state, keeping the embedding compact.

**Limits.** Chaotic systems have continuous spectra and no finite-dimensional invariant subspace, so finite Koopman models are approximations valid over a time horizon. Multiple attractors and fixed points cannot all be captured by one linear system in some coordinates. Learned embeddings can be unstable when rolled out far, like any learned dynamics.

## SINDy, Koopman or a neural surrogate?

| | SINDy | Koopman (EDMD / deep) | Neural surrogate (GNN, operator) |
| --- | --- | --- | --- |
| Output | A sparse, readable equation | A linear model in lifted coordinates | A black-box map |
| Data | Low-dimensional states (or reduced coordinates), derivatives | Trajectories | Many simulations of full fields |
| Strength | Interpretability, extrapolation if the terms are right | Linear analysis, prediction and control | Accuracy on high-dimensional fields and geometry |
| Weakness | Needs the right coordinates and library; noise in derivatives | Choosing or learning observables; chaotic dynamics | Little insight; needs data; extrapolates poorly |

## Code: recovering the Lorenz equations and a Koopman embedding

First, STLSQ on noisy measurements of the chaotic Lorenz system, with derivatives estimated by finite differences from the data and a library of all polynomials up to degree two. Second, a nonlinear system that becomes exactly linear in the observables (x₁, x₂, x₁²): a one-step linear model fitted by least squares (EDMD) forecasts it almost perfectly, while a linear model in the original coordinates (plain DMD) cannot.

```python
import itertools
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, s, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

dt = 0.002
t = np.arange(0, 20, dt)
X = solve_ivp(lorenz, (t[0], t[-1]), [-8.0, 8.0, 27.0], t_eval=t, rtol=1e-10, atol=1e-10).y.T
rng = np.random.default_rng(0)
X_noisy = X + rng.normal(0, 0.01, X.shape)                       # measurement noise
dXdt = np.gradient(X_noisy, dt, axis=0)                          # derivatives estimated from the data

def library(X, degree=2):
    """Candidate terms: 1, x, y, z, x^2, xy, ..., up to the given polynomial degree."""
    names, cols = ["1"], [np.ones(len(X))]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(X.shape[1]), d):
            names.append("".join("xyz"[i] for i in combo))
            cols.append(np.prod(X[:, combo], axis=1))
    return np.stack(cols, 1), names

def stlsq(Theta, dXdt, threshold=0.1, iters=10):
    """Sequentially thresholded least squares (Brunton, Proctor & Kutz 2016)."""
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        for k in range(dXdt.shape[1]):                           # refit each equation on its surviving terms
            keep = ~small[:, k]
            Xi[keep, k] = np.linalg.lstsq(Theta[:, keep], dXdt[:, k], rcond=None)[0]
    return Xi

Theta, names = library(X_noisy)
Xi = stlsq(Theta[5:-5], dXdt[5:-5], threshold=0.5)              # drop the ends, where gradients are one-sided
for k, lhs in enumerate(["dx/dt", "dy/dt", "dz/dt"]):
    print(f"{lhs} = " + " ".join(f"{Xi[i, k]:+.3f} {n}" for i, n in enumerate(names) if Xi[i, k] != 0))
print(f"{np.count_nonzero(Xi)} non-zero terms out of {Xi.size} candidates (true model: 7)")

# Koopman: dx1/dt = mu x1, dx2/dt = lam (x2 - x1^2) is exactly linear in the observables (x1, x2, x1^2).
mu, lam = -0.05, -1.0
def slow_manifold(t, s):
    return [mu * s[0], lam * (s[1] - s[0] ** 2)]
dt2 = 0.1
times = np.arange(0, 10 + 1e-9, dt2)
train = [solve_ivp(slow_manifold, (0, 10), x0, t_eval=times, rtol=1e-10).y.T for x0 in rng.uniform(-1.5, 1.5, (20, 2))]
test = solve_ivp(slow_manifold, (0, 10), [1.2, -0.8], t_eval=times, rtol=1e-10).y.T
lift = lambda S: np.column_stack([S[:, 0], S[:, 1], S[:, 0] ** 2])
for name, observables in [("linear model in (x1, x2)       ", lambda S: S), ("linear model in (x1, x2, x1^2)", lift)]:
    Z0 = np.vstack([observables(T[:-1]) for T in train])
    Z1 = np.vstack([observables(T[1:]) for T in train])
    K = np.linalg.lstsq(Z0, Z1, rcond=None)[0]                  # one-step linear operator: z_{k+1} = z_k K
    z = [observables(test[:1])[0]]
    for _ in range(len(test) - 1):
        z.append(z[-1] @ K)
    print(f"{name}: max error over a 100-step forecast {np.abs(np.array(z)[:, :2] - test).max():.2e}")
```

Output:

```
dx/dt = -9.999 x +9.999 y
dy/dt = +27.990 x -0.998 y -1.000 xz
dz/dt = -2.666 z +1.000 xy
7 non-zero terms out of 30 candidates (true model: 7)
linear model in (x1, x2)       : max error over a 100-step forecast 1.91e+00
linear model in (x1, x2, x1^2): max error over a 100-step forecast 4.96e-07
```

**What it shows.**

- SINDy keeps exactly the seven true terms out of 30 candidates and recovers the coefficients (σ = 10, ρ = 28, β = 8/3 ≈ 2.667) to within 0.2%, even with noisy data and finite-difference derivatives.
- The nonlinear system is impossible to forecast with a linear model in its own coordinates, but adding the single observable x₁² makes it exactly linear, and the least-squares Koopman matrix forecasts 100 steps with error 5 × 10⁻⁷. Finding such observables automatically, for systems where they are not known, is what deep Koopman models try to do.

## In an interview

**Thirty-second answer to "What is SINDy?"** "Sparse identification of nonlinear dynamics. You measure a state over time, estimate derivatives, build a library of candidate terms such as polynomials, and solve a sparse regression so that only a few terms survive, typically by sequentially thresholded least squares. The output is a readable differential equation; on the Lorenz system it recovers the exact seven terms from noisy data. It needs good coordinates, often POD coordinates for fields, and noise in the derivatives is its main weakness."

**Thirty-second answer to "What is the Koopman operator, and how is deep learning used to approximate it?"** "It advances observables, functions of the state, one step: Kg(x) = g(F(x)). It is linear even for nonlinear dynamics, but infinite-dimensional. If you find a few observables whose span is invariant, the dynamics are a matrix in those coordinates, so prediction, spectra and control become linear algebra. DMD uses the state itself, extended DMD a chosen dictionary, and deep Koopman models learn the observables with an autoencoder whose latent dynamics are a linear matrix, with an auxiliary network for continuous spectra."

**Likely follow-ups**

- *How does DMD relate to Koopman?* DMD is the Koopman approximation with linear observables.
- *How do you handle noisy derivatives in SINDy?* Smoothing or total-variation differentiation, the weak formulation, or ensembles.
- *Can Koopman models handle chaos?* Only approximately over finite horizons; chaotic systems have continuous spectra.

## References

- [Discovering governing equations from data by sparse identification of nonlinear dynamical systems (Brunton, Proctor & Kutz, PNAS 2016)](https://arxiv.org/abs/1509.03580)
- [Deep learning for universal linear embeddings of nonlinear dynamics (Lusch, Kutz & Brunton, Nature Communications 2018)](https://arxiv.org/abs/1712.09707)
- [Data-driven discovery of partial differential equations (PDE-FIND; Rudy et al.)](https://arxiv.org/abs/1609.06401)
- [Modern Koopman Theory for Dynamical Systems (Brunton, Budišić, Kaiser & Kutz)](https://arxiv.org/abs/2102.12086)
- [PySINDy](https://github.com/dynamicslab/pysindy) and [PyDMD](https://github.com/PyDMD/PyDMD)
- S. L. Brunton and J. N. Kutz, *Data-Driven Science and Engineering*, Cambridge University Press ([book website](https://databookuw.com/)).
