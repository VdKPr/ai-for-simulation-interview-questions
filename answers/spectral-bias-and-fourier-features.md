# Spectral Bias and Fourier Features

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** **Spectral bias** (also called the frequency principle) is the tendency of neural networks trained by gradient descent to learn the smooth, low-frequency parts of a function first and the sharp, high-frequency parts very slowly, if at all. Physical fields are full of high-frequency content where it matters most: boundary layers, shocks, stress concentrations, thin thermal layers, turbulence. A network with spectral bias reproduces the overall shape and smooths away the peaks, which is often the unsafe direction. **Fourier features** (also called positional encodings) counter it by mapping input coordinates through sines and cosines at many frequencies, γ(x) = [sin(2πBx), cos(2πBx)], before the network. That lets a plain MLP fit high-frequency detail quickly. They are needed when a network takes **coordinates as input** and must represent sharp or oscillatory fields (PINNs, neural fields, coordinate-based decoders, transformers on point clouds), and the frequency scale must be chosen: too low cannot fit the detail, too high fits noise and interpolates badly between samples.

**On this page**

- [Spectral bias](#spectral-bias)
- [Why it matters for physical fields](#why-it-matters-for-physical-fields)
- [Fourier features and positional encodings](#fourier-features-and-positional-encodings)
- [Code: a network that cannot learn high frequencies, and the fix](#code-a-network-that-cannot-learn-high-frequencies-and-the-fix)
- [Choosing the frequency scale](#choosing-the-frequency-scale)
- [Other remedies](#other-remedies)
- [In an interview](#in-an-interview)

## Spectral bias

Decompose a target function into frequency components. Rahaman et al. (2019) showed that deep ReLU networks fit the low-frequency components early in training and the high-frequency components much later; Xu and colleagues called the same observation the *frequency principle*. The neural tangent kernel (NTK) view makes it precise: in the regime where training behaves like kernel regression, the error along each eigenfunction of the NTK decays at a rate proportional to its eigenvalue, and for standard networks with coordinate inputs the eigenvalues fall rapidly with frequency. High-frequency components therefore converge orders of magnitude more slowly.

A picture of what training does to a field with a sharp feature:

```
target:        ‾‾‾‾╲___╱‾‾‾‾‾‾‾‾‾‾╲/‾‾‾‾‾      (smooth trend + sharp dip)
early epochs:  ‾‾‾‾╲___╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾      trend learned, dip missing
late epochs:   ‾‾‾‾╲___╱‾‾‾‾‾‾‾‾‾‾╲_/‾‾‾‾      dip appears slowly, and shallower
```

Spectral bias is not all bad: it is part of why networks generalize well on smooth problems instead of fitting noise. The trouble starts when the high frequencies are physics, not noise.

## Why it matters for physical fields

- **Boundary layers and shear layers:** velocity changes from zero to free-stream over a tiny distance.
- **Shocks and contact discontinuities** in compressible flow.
- **Stress concentrations** at fillets and holes: the peak that decides the safety factor is a sharp, local feature. See [Predicting a safety factor with a surrogate](von-mises-stress-and-yield.md#predicting-a-safety-factor-with-a-surrogate).
- **Multi-scale and oscillatory solutions:** turbulence, wave propagation, high-frequency vibration modes, composite microstructures.
- **Long rollouts:** autoregressive models that neglect small-scale components accumulate error there. See [Losing small scales and PDE-Refiner](autoregressive-rollouts.md#losing-small-scales-and-pde-refiner).

In each case, a spectrally biased network gives a plausible, smooth field with the right overall shape and the wrong peaks. Because averaged losses are dominated by the large smooth regions, the training loss can look fine.

**For PINNs** the problem is sharper still: the network is fitted to a PDE residual rather than data, and high-frequency or multi-scale solutions are among the classic PINN failure modes. See [High-frequency and multi-scale solutions](pinn-training-failures.md#high-frequency-and-multi-scale-solutions).

## Fourier features and positional encodings

**The idea.** Instead of feeding the raw coordinate x to the network, feed a set of sines and cosines of it:

```math
\gamma(\mathbf{x}) = \big[\sin(2\pi B\mathbf{x}),\; \cos(2\pi B\mathbf{x})\big]
```

where B is a matrix of frequencies. Tancik et al. (2020) showed that this mapping turns the network's effective kernel into a stationary kernel whose bandwidth is controlled by B, so a plain MLP can learn high-frequency functions in low-dimensional domains. Two common choices:

- **Random Fourier features:** entries of B drawn from a Gaussian with standard deviation σ. σ sets the frequency scale.
- **Deterministic (positional) encodings:** frequencies on a geometric ladder, 2⁰, 2¹, …, 2^(L−1), as in transformers and neural radiance fields.

**When you need them:**

- A network maps **coordinates to field values**: PINNs, neural fields (implicit neural representations), the decoders of neural operators that are queried at arbitrary points, coordinate-based transformers on point clouds. See [Neural fields](geometry-representations.md#neural-fields).
- The target has **sharp or oscillatory structure** relative to the domain size.
- Positions must be distinguishable at fine resolution, as in transformers whose attention should depend on relative position. See [Encoding positions and boundary conditions](transformers-for-physics.md#encoding-positions-and-boundary-conditions).

**When you may not:** grid-based CNNs and U-Nets, and graph networks that use relative edge features, see local structure directly through their architecture; smooth targets and scalar outputs from design parameters do not need them.

A related idea is the SIREN architecture (Sitzmann et al., 2020), which uses sine activations throughout the network with a careful initialization; it also represents fine detail and has well-behaved derivatives, which helps PINN-like applications.

## Code: a network that cannot learn high frequencies, and the fix

Fit f(x) = sin(2πx) + 0.5 sin(2π·20x) on [0, 1] with a ReLU MLP, with and without random Fourier features, and measure how much of each frequency is still missing from the fit.

```python
import math
import torch

torch.manual_seed(0)
x = torch.linspace(0, 1, 512).unsqueeze(1)
y = torch.sin(2 * math.pi * x) + 0.5 * torch.sin(2 * math.pi * 20 * x)    # low + high frequency

class FourierFeatures(torch.nn.Module):
    """gamma(x) = [sin(2 pi B x), cos(2 pi B x)] with random frequencies B ~ N(0, sigma^2)."""
    def __init__(self, n_features=64, sigma=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(1, n_features) * sigma)
    def forward(self, x):
        z = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

def mlp(n_in):
    return torch.nn.Sequential(torch.nn.Linear(n_in, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 1))

def residual_amplitude(residual, k):
    """Amplitude of the sin(2 pi k x) component left in the residual."""
    return (2 * (residual * torch.sin(2 * math.pi * k * x)).mean()).abs().item()

missing = {}
for name, model in [("plain MLP", mlp(1)),
                    ("Fourier features + MLP", torch.nn.Sequential(FourierFeatures(), mlp(128)))]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 2001):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (200, 2000):
            with torch.no_grad():
                r = y - model(x)
            missing[(name, step)] = residual_amplitude(r, 20)
            print(f"{name:24s} step {step:4d}: error left at frequency 1 = {residual_amplitude(r, 1):.3f}, "
                  f"at frequency 20 = {residual_amplitude(r, 20):.3f}")
```

Output:

```
plain MLP                step  200: error left at frequency 1 = 0.007, at frequency 20 = 0.490
plain MLP                step 2000: error left at frequency 1 = 0.004, at frequency 20 = 0.492
Fourier features + MLP   step  200: error left at frequency 1 = 0.000, at frequency 20 = 0.000
Fourier features + MLP   step 2000: error left at frequency 1 = 0.000, at frequency 20 = 0.000
```

The plain network learns the low-frequency wave almost immediately and, after 2,000 steps, has still learned essentially none of the 0.5-amplitude high-frequency wave. With Fourier features, both are fitted within 200 steps.

## Choosing the frequency scale

The frequency scale σ (or the number of octaves L) is a real hyperparameter:

- **Too low:** the encoding cannot express the fine detail; the result looks like the plain network.
- **Too high:** the network can fit anything, including noise, and *interpolates badly between training points*: the prediction oscillates between samples. Tancik et al. showed exactly this trade-off between underfitting and overfitting as σ varies.
- **Rule of thumb:** match σ to the highest frequency that is physically present in the field, relative to coordinates scaled to a unit range, and tune it on held-out points or held-out designs.
- **Multi-scale solutions** may need several scales at once. For PINNs, Wang, Wang and Perdikaris (2021) proposed multi-scale Fourier feature architectures that combine several σ values for problems with both low and high frequencies.

Always scale coordinates to a fixed range (for example [−1, 1] or [0, 1]) before applying the encoding, otherwise the meaning of σ changes with the units. See [Normalizing fields and coordinates](nondimensionalization-and-normalization.md#normalizing-fields-and-coordinates).

## Other remedies

- **Architectures with built-in multi-scale structure:** U-Nets, multiscale GNNs, Fourier neural operators (which explicitly carry frequency modes), multigrid-inspired designs. See [Multiscale GNNs](multiscale-gnns.md) and [Fourier Neural Operator](fourier-neural-operator.md).
- **Loss design:** weight the loss towards high-gradient regions, add gradient (Sobolev) terms, or add an explicit loss on the peak value.
- **Refinement and diffusion-style models:** PDE-Refiner improves high-frequency accuracy in rollouts through a multistep refinement process inspired by diffusion models. See [Losing small scales and PDE-Refiner](autoregressive-rollouts.md#losing-small-scales-and-pde-refiner).
- **Adaptive sampling:** put more training points (or collocation points) where the solution varies quickly. See [Collocation points](pinns.md#collocation-points).
- **Domain decomposition:** smaller subdomains turn high frequencies into low ones locally (for example finite-basis PINNs).

## In an interview

**Thirty-second answer to "What is spectral bias, and why does it matter for physical fields?"** "Networks trained with gradient descent learn low-frequency components first and high-frequency ones much more slowly; in the neural tangent kernel picture, high-frequency eigenfunctions have small eigenvalues and converge slowly. Physical fields have sharp features exactly where engineers care, such as boundary layers, shocks and stress peaks, so a spectrally biased surrogate gives a smooth field with under-predicted peaks while the average loss looks fine."

**Thirty-second answer to "What are Fourier features, and when do you need them?"** "Map input coordinates through sines and cosines at many frequencies before the network. That changes the network's effective kernel so it can learn high frequencies quickly. You need them when a network takes coordinates as input and must represent sharp or oscillatory fields, as in PINNs, neural fields and point-query decoders. The frequency scale must be tuned: too low underfits, too high overfits and oscillates between samples."

**Likely follow-ups**

- *Is spectral bias always bad?* No: it helps generalization on smooth targets and suppresses noise; it is a problem when high frequencies are real physics.
- *How would you choose σ?* Scale coordinates to a unit range, match σ to the highest physical frequency, and tune on held-out data; use multi-scale features for multi-scale solutions.
- *Do graph networks need Fourier features?* Less often: relative edge features and local message passing already see local structure, but absolute-position inputs still benefit.

## References

- [On the Spectral Bias of Neural Networks (Rahaman et al., 2019)](https://arxiv.org/abs/1806.08734)
- [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains (Tancik et al., 2020)](https://arxiv.org/abs/2006.10739)
- [On the eigenvector bias of Fourier feature networks: from regression to solving multi-scale PDEs with physics-informed neural networks (Wang, Wang & Perdikaris, 2021)](https://arxiv.org/abs/2012.10047)
- [Implicit Neural Representations with Periodic Activation Functions (SIREN, Sitzmann et al., 2020)](https://arxiv.org/abs/2006.09661)
- [When and why PINNs fail to train: A neural tangent kernel perspective (Wang, Yu & Perdikaris)](https://arxiv.org/abs/2007.14527)
