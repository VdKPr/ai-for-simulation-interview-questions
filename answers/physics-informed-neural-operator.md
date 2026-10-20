# Physics-Informed Neural Operator (PINO)

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** A physics-informed neural operator (PINO), introduced by Li et al., combines the two main ideas of scientific machine learning. Like a **neural operator**, it learns the solution operator of a whole family of PDE problems (from coefficients, initial or boundary conditions to solutions), so a new instance costs one forward pass. Like a **PINN**, its training loss includes the **PDE residual**, so it needs little or no solution data. PINO uses a Fourier neural operator as its backbone and combines **coarse-resolution data** with the PDE constraint imposed at a **higher resolution**, which gives accurate predictions even beyond the training data's resolution. It can also be trained from the PDE alone, where plain PINNs fail on hard problems such as Kolmogorov flow, and its prediction for a specific instance can be refined at test time by fine-tuning on that instance's PDE residual.

**On this page**

- [PINN, neural operator and PINO compared](#pinn-neural-operator-and-pino-compared)
- [How PINO is trained](#how-pino-is-trained)
- [Computing the residual on an operator's output](#computing-the-residual-on-an-operators-output)
- [Code: a spectral derivative and a physics loss](#code-a-spectral-derivative-and-a-physics-loss)
- [When PINO is the right tool](#when-pino-is-the-right-tool)
- [Related approaches](#related-approaches)
- [In an interview](#in-an-interview)

## PINN, neural operator and PINO compared

| | PINN | Neural operator (data-driven) | PINO |
| --- | --- | --- | --- |
| Learns | One solution u(x, t) | The solution operator: inputs → solution | The solution operator |
| Training signal | PDE residual (+ boundary, initial, optional data) | Pairs of (input, solution) from a solver | Data pairs and/or the PDE residual |
| Data needed | None, or sparse measurements | Many solved instances | Fewer solved instances, possibly coarse; or none |
| New instance | Retrain (minutes to hours) | One forward pass | One forward pass, optionally plus fine-tuning |
| Physics consistency | Enforced by the loss | Only as good as the data | Enforced by the loss |
| Typical weakness | Hard optimization; per-instance training | Data hungry; can violate physics | Needs the PDE and a way to compute its residual on the output |

See [PINNs](pinns.md) and [Neural operators](neural-operators.md).

## How PINO is trained

**Phase 1: operator learning.** Train the operator G_θ over a distribution of problem instances a (coefficient fields, initial conditions) with a combined loss:

```math
\mathcal{L}(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \big\lVert G_\theta(a_i) - u_i \big\rVert^2}_{\text{data, possibly coarse}} \;+\; \lambda\,\underbrace{\mathbb{E}_{a}\Big[\big\lVert \mathcal{N}\big(G_\theta(a), a\big)\big\rVert^2\Big]}_{\text{PDE residual, evaluated at higher resolution}}
```

- The **data term** uses whatever solved examples exist, which can be on a coarse grid (cheap to generate).
- The **physics term** evaluates the residual of the predicted solution, and it can be evaluated on a finer grid than the data and on **new input instances without any solution data**, because only the PDE is needed.
- Li et al. report that PINO approximates the ground-truth operator for many standard PDE families and shows no degradation in accuracy even in zero-shot super-resolution, predicting beyond the resolution of the training data.

**Phase 2 (optional): instance-wise fine-tuning.** For a specific new instance, start from the trained operator's prediction and fine-tune the weights on that instance's PDE residual alone. The operator provides an excellent initialization, which avoids the optimization difficulties of training a PINN from scratch, and the residual pushes the prediction towards the exact solution for that instance.

**Training with no data.** Because the residual term needs only the PDE, PINO can be trained purely on physics. The paper reports success in settings where previous PINN approaches failed due to optimization challenges, including multi-scale dynamic systems such as Kolmogorov flows.

## Computing the residual on an operator's output

A PINN differentiates its network with respect to input coordinates by automatic differentiation. An FNO outputs values on a grid, so PINO computes derivatives in other ways:

- **Spectral (Fourier) differentiation** on periodic grids: transform with the FFT, multiply each mode by (ik)ⁿ, transform back. Very accurate for smooth periodic fields and cheap, since FNO already works in Fourier space.
- **Finite differences** on the output grid: simple, works for non-periodic domains, less accurate.
- **Automatic differentiation** through the operator when it is queried at arbitrary points (exact, but more expensive).

Boundary conditions must be handled consistently with the chosen derivative (spectral derivatives assume periodicity).

## Code: a spectral derivative and a physics loss

```python
import math
import torch

def spectral_derivative(u, L=2 * math.pi, order=1):
    """d^order u / dx^order of periodic samples u[..., n] on [0, L), computed with the FFT.
    This is how a residual can be evaluated on a gridded operator output (e.g. an FNO) cheaply."""
    n = u.shape[-1]
    k = torch.fft.rfftfreq(n, d=L / n, dtype=u.dtype) * 2 * math.pi   # angular wavenumbers
    return torch.fft.irfft((1j * k) ** order * torch.fft.rfft(u), n=n)

n = 128
x = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
u = torch.sin(3 * x) + 0.5 * torch.cos(5 * x)
u_xx = spectral_derivative(u, order=2)
exact = -9 * torch.sin(3 * x) - 12.5 * torch.cos(5 * x)
deriv_error = (u_xx - exact).abs().max().item()
print(f"max error of the spectral second derivative: {deriv_error:.1e}")

# A physics loss for the steady equation -u'' = f on the periodic grid, as used in physics-informed operator training:
f = 9 * torch.sin(3 * x) + 12.5 * torch.cos(5 * x)
physics_loss = ((-spectral_derivative(u, order=2) - f) ** 2).mean()
print(f"physics loss of the exact solution: {physics_loss:.1e}")
```

Output:

```
max error of the spectral second derivative: 4.6e-12
physics loss of the exact solution: 2.0e-24
```

In a PINO, u would be the operator's predicted field for a batch of input instances, and the physics loss would be added to (or replace) the data loss. Note the `dtype=u.dtype`: computing the wavenumbers in single precision alone limits the accuracy to about 10⁻⁶.

## When PINO is the right tool

**Good fit:**

- The PDE is known and trusted, and a **family** of instances must be solved quickly (many coefficient fields, initial conditions or forcing terms).
- Solution data is expensive or available only at **coarse resolution**.
- Predictions must be **physically consistent** (small residual), not just close to the training data.
- You can afford test-time fine-tuning for the cases that matter most.

**Poor fit:**

- The physics is incomplete or uncertain (then the residual pushes towards the wrong model).
- Complex 3D geometries where evaluating residuals on the operator's output is hard (no regular grid, complex boundaries); geometry-aware operators help but residual evaluation remains the bottleneck.
- Problems where plenty of high-quality data exists and the physics adds little.

## Related approaches

- **Physics-informed DeepONet** (Wang, Wang and Perdikaris, 2021): adds a PDE residual to DeepONet training, computed by automatic differentiation through the trunk network's coordinate inputs. See [DeepONet](deeponet.md).
- **Parametric PINNs:** add the problem parameters as network inputs; simpler, but limited to low-dimensional parameter spaces.
- **Hybrid solvers:** let a classical solver compute the physics and learn only corrections. See [Hybrid solvers](differentiable-physics-and-hybrid-solvers.md).

## In an interview

**Thirty-second answer to "What is a physics-informed neural operator?"** "It is a neural operator, built on FNO, trained with both data and the PDE residual. Like an operator it learns the map from problem inputs to solutions across a whole family, so new instances take one forward pass; like a PINN it uses the residual, so it needs little or even no solution data. PINO combines coarse-resolution data with the residual at higher resolution, which gives accurate predictions beyond the data's resolution, and it can be fine-tuned on a single instance's residual at test time. It works where plain PINNs fail on hard problems like Kolmogorov flow, but it needs the PDE and a way to evaluate its residual on the output."

**Likely follow-ups**

- *How is the residual computed on an FNO output?* Spectral derivatives on periodic grids, finite differences, or autograd if the operator is queried at points.
- *Why does fine-tuning at test time help?* The operator gives a near-correct initialization, so residual-only optimization converges where a PINN from scratch would struggle.
- *PINO vs PINN for a single inverse problem?* A PINN is simpler for one instance; PINO pays off for families of instances.

## References

- [Physics-Informed Neural Operator for Learning Partial Differential Equations (Li et al.)](https://arxiv.org/abs/2111.03794)
- [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al.)](https://arxiv.org/abs/2010.08895)
- [Learning the solution operator of parametric partial differential equations with physics-informed DeepONets (Wang, Wang & Perdikaris)](https://arxiv.org/abs/2103.10974)
- [neuraloperator library (includes FNO and related operators)](https://github.com/neuraloperator/neuraloperator)
