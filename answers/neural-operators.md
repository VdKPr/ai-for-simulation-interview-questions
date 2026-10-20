# Neural Operators

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A neural operator learns a map between **functions**: from an input function (a coefficient field, a forcing, an initial or boundary condition, a geometry) to the solution function. It is trained on many solved instances from a numerical solver and then solves a new instance in one forward pass. Its layers are designed to approximate continuous operators, typically **integral kernels** ∫κ(x, y)v(y)dy evaluated by FFTs (FNO), neighbourhood sums (graph neural operators), attention (transformer operators) or a branch-trunk decomposition (DeepONet), so the same weights can be applied to inputs on different discretizations. That is the difference from a **PINN**, which represents *one* solution and is trained per instance by minimizing the PDE residual, and from ordinary regression between fixed-length vectors. "Discretization invariance" holds mechanically (the model runs on new grids), but accuracy on new resolutions must still be tested. Operators need hundreds to thousands of solved examples, so data generation is usually the main cost.

**On this page**

- [PINN vs neural operator](#pinn-vs-neural-operator)
- [What operator learning is](#what-operator-learning-is)
- [The neural-operator layer: a learned integral kernel](#the-neural-operator-layer-a-learned-integral-kernel)
- [Code: one kernel-integral layer at four resolutions](#code-one-kernel-integral-layer-at-four-resolutions)
- [Discretization invariance](#discretization-invariance)
- [The main families](#the-main-families)
- [Graph neural operators](#graph-neural-operators)
- [Convolutional neural operators](#convolutional-neural-operators)
- [Neural operators for time-dependent problems](#neural-operators-for-time-dependent-problems)
- [How much data a neural operator needs](#how-much-data-a-neural-operator-needs)
- [Generalization](#generalization)
- [In an interview](#in-an-interview)

## PINN vs neural operator

| | PINN | Neural operator |
| --- | --- | --- |
| What it represents | **One solution** u(x, t) for fixed coefficients, boundary conditions and geometry | **The solution operator** G: (input function) → (solution function) for a whole family of problems |
| Inputs to the network | Coordinates (x, t) | An input function (sampled on a grid, mesh or points), and often query coordinates |
| Training signal | PDE residual, boundary and initial conditions, optional sparse data | Input-output pairs from a solver, optionally plus a PDE residual (PINO) |
| Data needed | None (forward) or sparse measurements (inverse) | Hundreds to thousands of solved instances |
| A new problem instance | Train again (minutes to hours) | One forward pass (milliseconds to seconds) |
| Strongest use | Inverse problems, data assimilation | Many-query work: design exploration, optimization, uncertainty studies, real-time use |

**The one-line version:** a PINN *solves* an equation; a neural operator *learns a solver* for a family of equations.

**A combination exists:** the physics-informed neural operator (PINO) trains an operator with both data and PDE residuals. See [Physics-informed neural operator](physics-informed-neural-operator.md).

## What operator learning is

**Ordinary supervised learning** learns a function between finite-dimensional vectors: ℝⁿ → ℝᵐ, with n and m fixed by the data format.

**Operator learning** learns a map between *function spaces*. Example, steady flow in porous media (Darcy flow): the input is the permeability field a(x) over the domain, the output is the pressure field u(x) that solves −∇·(a∇u) = f. The "true" input and output are functions with infinitely many values; any grid is just one way of sampling them.

```
input function a(x)               solution operator G                output function u(x)
(permeability, initial            (what a PDE solver does,           (pressure, later state,
 condition, geometry, BCs)  ───▶  learned from examples)      ───▶   displacement, ...)
```

Two consequences shape the architectures:

1. **The model should not depend on how the functions are sampled.** The same physical input on a 64 × 64 grid or a 256 × 256 grid, or on an unstructured mesh, should produce (approximately) the same output.
2. **The output may be needed anywhere,** not only at the input points, so many operators can be *queried* at arbitrary coordinates.

Why it matters commercially: a trained operator replaces many solver runs. Training costs the price of the dataset; each later query is almost free.

## The neural-operator layer: a learned integral kernel

Kovachki et al. formulated a general neural-operator layer that maps a function v_t to a new function v_{t+1}:

```math
v_{t+1}(x) = \sigma\!\left( W\,v_t(x) + \int_D \kappa_\theta(x, y)\, v_t(y)\, dy \right)
```

- W v_t(x) is a pointwise linear map (a local path);
- the integral with a learned kernel κ_θ is a *non-local* path: every point can depend on every other point, as in the solution of an elliptic PDE (the Green's function of a linear PDE is exactly such a kernel);
- σ is a pointwise nonlinearity.

A full operator lifts the input to many channels, applies several such layers, and projects back to the output. **The families differ in how they compute the integral:**

- **FNO:** assume a translation-invariant kernel κ(x − y), so the integral is a convolution, computed in Fourier space with an FFT, keeping only low modes. See [Fourier Neural Operator](fourier-neural-operator.md).
- **Graph neural operator:** approximate the integral by a sum over neighbouring points within a radius, like message passing. See [below](#graph-neural-operators).
- **Transformer operators:** attention computes a data-dependent kernel; linear-attention variants are a Galerkin-type approximation of an integral operator. See [Is attention a neural operator?](transformers-for-physics.md#is-attention-a-neural-operator).
- **DeepONet:** a different construction, a sum of learned basis functions weighted by coefficients computed from the input. See [DeepONet](deeponet.md).

## Code: one kernel-integral layer at four resolutions

The defining property in its simplest form: an integral evaluated from function samples converges as the sampling gets finer, and the same "layer" accepts any number and placement of input points. (Here the kernel is fixed; in a neural operator it is learned.)

```python
import numpy as np

rng = np.random.default_rng(0)
kappa = lambda x, y: np.exp(-((x[:, None] - y[None, :]) ** 2) / 0.02)   # a (here fixed) kernel kappa(x, y)
v = lambda y: np.sin(2 * np.pi * y)                                     # input function on [0, 1]
x_query = np.array([0.1, 0.35, 0.6, 0.85])                              # where the output is wanted

def kernel_integral(n_points):
    """(K v)(x) = integral kappa(x, y) v(y) dy, approximated from v sampled at n random points:
    the same 'layer' accepts any number and placement of input points."""
    y = rng.random(n_points)
    return (kappa(x_query, y) * v(y)[None, :]).mean(axis=1)            # Monte Carlo quadrature

y_fine = np.linspace(0, 1, 200001)
reference = np.trapezoid(kappa(x_query, y_fine) * v(y_fine)[None, :], y_fine, axis=1)
errors = []
for n in [50, 500, 5000, 50000]:
    errors.append(np.abs(kernel_integral(n) - reference).max())
    print(f"{n:6d} input points: max error vs fine quadrature = {errors[-1]:.4f}")
```

Output:

```
    50 input points: max error vs fine quadrature = 0.0543
   500 input points: max error vs fine quadrature = 0.0239
  5000 input points: max error vs fine quadrature = 0.0031
 50000 input points: max error vs fine quadrature = 0.0018
```

A standard MLP on a fixed-length input vector cannot even accept 500 values if it was built for 50. An operator layer can, and its output converges as the discretization is refined.

## Discretization invariance

**The claim.** Because each layer approximates a continuous operator, the same learned parameters can be applied to inputs on different grids or meshes, and the outputs converge to the continuum operator as the discretization is refined. Kovachki et al. call this property *discretization convergence* (often loosely called discretization or resolution invariance). GINO, for example, is described as discretization-convergent.

**What holds in practice:**

- The model **runs** at resolutions it was not trained on. That is real and useful (for example, training on cheap coarse data and predicting on finer grids).
- Accuracy at new resolutions is **not guaranteed**:
  - **Aliasing.** Pointwise nonlinearities create high frequencies that the discrete grid cannot represent, so the discrete model is not exactly the continuous operator, and results shift with resolution. Bartolucci et al. (2023) formalized this as *operator aliasing* and proposed representation-equivalent neural operators (ReNO); the convolutional neural operator was designed to avoid it.
  - **Interpolation of inputs** to the model's resolution changes the input itself.
  - **Physics not in the data.** A model trained on data that does not resolve small scales cannot invent them at a higher resolution; "zero-shot super-resolution" only adds smooth interpolation, not new physics.
- **Test it explicitly:** evaluate the same test cases at several resolutions (and, for meshes, several meshings) and report the error at each.

## The main families

| Family | Key idea | Inputs and outputs | Notes |
| --- | --- | --- | --- |
| [DeepONet](deeponet.md) | Branch network (input function at sensors) × trunk network (query point) | Input at fixed sensor locations; output anywhere | Simple; flexible output; fixed input sensors |
| [FNO](fourier-neural-operator.md) | Global convolution in Fourier space, truncated modes | Regular grids | Fast and accurate on grids; Geo-FNO and GINO extend it to irregular geometry |
| Graph neural operator, GINO | Kernel integral over neighbourhoods; GINO maps irregular points to a latent grid, applies FNO, maps back | Point clouds and meshes | 3D industrial geometry |
| CNO | Convolutions with alias-free up- and down-sampling | Regular grids | Robust on multi-scale problems |
| Transformer operators (Galerkin transformer, OFormer, GNOT, Transolver, UPT, AB-UPT) | Attention as a learned, data-dependent integral kernel; linear attention or latent tokens for scale | Points, meshes, arbitrary queries | Current state of the art for large 3D aerodynamics ([Transformers for physics](transformers-for-physics.md)) |
| Physics-informed operators (PINO, physics-informed DeepONet) | Any of the above, trained with PDE residuals too | As backbone | Less data needed ([PINO](physics-informed-neural-operator.md)) |

## Graph neural operators

**The graph neural operator (GNO)** (Li et al., 2020) approximates the kernel integral by a sum over points y within a radius r of each point x:

```math
\int_D \kappa(x, y)\, v(y)\, dy \;\approx\; \frac{1}{|N(x)|}\sum_{y \in N(x)} \kappa_\theta\big(x, y, a(x), a(y)\big)\, v(y)
```

The kernel κ_θ is a small neural network of the two positions (and possibly the input function values there). This is message passing on a radius graph, with an interpretation as a Monte Carlo approximation of an integral, which is what makes it resolution-agnostic: sampling more points refines the same integral instead of changing the model. The cost grows with the number of neighbours, so the radius limits how far information travels per layer.

**The geometry-informed neural operator (GINO)** (Li et al., 2023) combines GNO and FNO for large 3D problems with varying geometry:

1. a GNO **encoder** maps the input point cloud (with a signed distance function describing the shape) onto a **regular latent grid**;
2. an **FNO** processes the latent grid efficiently, with global receptive field;
3. a GNO **decoder** maps back to the output points, for example the car surface.

The authors generated a 3D vehicle aerodynamics dataset with Reynolds numbers up to five million and trained GINO to predict surface pressure from only about five hundred training examples. See [Geometry representations](geometry-representations.md) for the role of the signed distance function.

## Convolutional neural operators

The **convolutional neural operator (CNO)** (Raonić et al., 2023) shows that convolution-based architectures, long believed to be inconsistent in function space, can be made into genuine operators. The key is to **preserve the continuous nature of the functions even in the discretized implementation**:

- work with band-limited functions and up- and down-sampling filters that respect the band limit;
- apply the pointwise nonlinearity on an upsampled representation and filter back down, so that the high frequencies it creates do not alias;
- arrange the layers in a U-Net-like encoder-decoder for multiple scales.

The paper proves a universality theorem for CNOs and introduces a benchmark suite of diverse PDEs with possibly multi-scale solutions, on which CNOs significantly outperformed baselines. In practice CNOs are a strong choice on regular grids, especially for problems with sharp or multi-scale features, and they connect the well-developed toolbox of CNNs and U-Nets to operator learning.

## Neural operators for time-dependent problems

Four common ways to handle time:

1. **Autoregressive stepping.** Learn the operator from the state at time t to the state at t + Δt and apply it repeatedly. Flexible and the most common, but errors accumulate and rollouts can drift or blow up; noise injection, the pushforward trick, temporal bundling and refinement methods address that. See [Autoregressive rollouts](autoregressive-rollouts.md).
2. **Space-time operators.** Treat time as another dimension and predict a whole window at once, for example an FNO with 3D convolutions over (x, y, t), as in the original FNO paper's Navier–Stokes experiments, compared with a 2D FNO stepping recurrently in time. No accumulation within the window, but memory grows with the window length.
3. **Time as an input.** Predict u(x, t) directly for any requested t, by conditioning the operator on t (Poseidon uses time-conditioned layer norms for continuous-in-time evaluation). Good for queries at arbitrary times; must still learn the whole trajectory.
4. **Latent dynamics.** Encode the state into a compact latent representation, evolve it in time there (with a learned step or a neural ODE), and decode only when needed. UPT, for example, propagates dynamics in latent space. See [Neural ODEs](neural-odes.md).

For crash and other transient structural problems, a recent study compared one-shot, time-conditional and autoregressive strategies and found the one-shot approach most accurate and cheapest. See [Temporal strategies](crash-simulation-surrogates.md#temporal-strategies-one-shot-time-conditional-autoregressive).

## How much data a neural operator needs

- **Typical scale:** hundreds to thousands of solved instances. The original FNO experiments used about a thousand training pairs per problem (and ten thousand for the hardest Navier–Stokes setting); industrial 3D datasets have a few hundred to a few thousand designs (GINO's car experiments used about five hundred).
- **It depends on** the dimension and variability of the inputs (a few parameters versus free-form geometry), the complexity of the solution (smooth versus multi-scale), the accuracy needed and the architecture's inductive biases.
- **Where it comes from:** a numerical solver run on a planned design of experiments, which is usually the dominant cost of the project. See [Generating simulation training data](generating-simulation-training-data.md).
- **How to need less:**
  - add physics losses (PINO, physics-informed DeepONet);
  - start from a pretrained model and fine-tune ([Foundation models](foundation-models-for-physics.md));
  - combine cheap low-fidelity with few high-fidelity runs ([Multi-fidelity learning](multi-fidelity-learning.md));
  - choose new simulations actively ([Active learning](design-of-experiments-and-active-learning.md#active-learning-and-acquisition-functions));
  - build in symmetries and non-dimensionalize.
- **Measure it:** a learning curve (error against number of training instances) on held-out data. See [the learning-curve example](generating-simulation-training-data.md#code-a-learning-curve-and-the-label-noise-floor).

## Generalization

- Reliable for inputs drawn from the same distribution as the training data.
- Not automatically reliable for new boundary-condition types, new geometry families, new parameter ranges or new physical regimes, however elegant the function-space formulation.
- Report in-distribution and out-of-distribution errors separately, and pair the model with out-of-distribution detection in use. See [Out-of-distribution detection and generalization](out-of-distribution-and-generalization.md).

## In an interview

**Thirty-second answer to "What is the difference between a PINN and a neural operator?"** "A PINN represents one solution: its inputs are coordinates, it is trained by minimizing the PDE residual for one fixed problem, and a new geometry or boundary condition means training again. A neural operator learns the solution operator for a whole family: it maps an input function, such as a coefficient field or geometry, to the solution function, is trained on many solver examples, and answers a new instance in one forward pass. PINNs suit inverse problems; operators suit many-query design work. PINO combines both."

**Thirty-second answer to "What is operator learning?"** "Learning maps between function spaces instead of between fixed-size vectors: for example from a permeability field to a pressure field. The layers approximate integral operators, with kernels computed via FFTs, neighbourhood sums or attention, so the model is defined independently of the discretization and can be applied on different grids, and often queried anywhere."

**Likely follow-ups**

- *Does discretization invariance hold?* The model runs at new resolutions, but aliasing, input interpolation and unresolved physics limit accuracy; test several resolutions.
- *How do you handle time?* Autoregressive stepping, space-time windows, time-conditioning or latent dynamics, each with its trade-offs.
- *How much data?* Hundreds to thousands of solved instances; physics losses, pretraining and multi-fidelity reduce it.

## References

- [Neural Operator: Learning Maps Between Function Spaces (Kovachki et al., JMLR)](https://arxiv.org/abs/2108.08481)
- [Neural Operator: Graph Kernel Network for Partial Differential Equations (Li et al.)](https://arxiv.org/abs/2003.03485)
- [Geometry-Informed Neural Operator for Large-Scale 3D PDEs (GINO, Li et al.)](https://arxiv.org/abs/2309.00583)
- [Convolutional Neural Operators for robust and accurate learning of PDEs (Raonić et al.)](https://arxiv.org/abs/2302.01178)
- [Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning (Bartolucci et al.)](https://arxiv.org/abs/2305.19913)
- [From Theory to Application: A Practical Introduction to Neural Operators](https://arxiv.org/abs/2503.05598)
- [ETH Zürich AISE: Introduction to Operator Learning Part 1 (video)](https://www.youtube.com/watch?v=yhHhMmiNl_g) and [Part 2 (video)](https://www.youtube.com/watch?v=lEUgPvDi5O8)
- [neuraloperator library](https://github.com/neuraloperator/neuraloperator)
