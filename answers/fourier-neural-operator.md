# Fourier Neural Operator (FNO)

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The Fourier Neural Operator (Li et al., 2020) learns the solution operator of a PDE family on a regular grid. It **lifts** the input function to many channels with a pointwise layer, applies a stack of **Fourier layers**, and **projects** back to the output. Each Fourier layer takes the FFT of its input, multiplies the lowest k Fourier modes by learned complex weights, sets the higher modes to zero, takes the inverse FFT, adds a pointwise linear path, and applies a nonlinearity. That is a global convolution with a learned kernel, computed in O(N log N). Keeping only a few modes makes the weights independent of the grid resolution, reduces parameters and regularizes, because smooth PDE solutions concentrate their energy in low frequencies; the price is difficulty with sharp fronts and shocks. FNO assumes uniform grids and periodicity, so complex geometry needs extensions: **Geo-FNO** learns a deformation from the physical domain to a uniform latent grid, and **GINO** uses graph neural operators to map irregular points onto a latent grid and back.

**On this page**

- [Architecture](#architecture)
- [The Fourier layer in detail](#the-fourier-layer-in-detail)
- [Why FNO keeps only a few Fourier modes](#why-fno-keeps-only-a-few-fourier-modes)
- [Code: what mode truncation does to smooth and sharp fields](#code-what-mode-truncation-does-to-smooth-and-sharp-fields)
- [Limitations, Geo-FNO and GINO](#limitations-geo-fno-and-gino)
- [Variants](#variants)
- [Code: a 1D Fourier layer](#code-a-1d-fourier-layer)
- [In an interview](#in-an-interview)

## Architecture

```
a(x) ──▶ P (lift, pointwise) ──▶ [Fourier layer] × L ──▶ Q (project, pointwise) ──▶ u(x)
 input        to `width`                 │                      to output
 function     channels                   │                      channels
                                         ▼
          v ──┬──▶ FFT ──▶ keep k lowest modes, multiply by R ──▶ inverse FFT ──┐
              │                                                                 (+) ──▶ σ ──▶ v'
              └──▶ W (pointwise linear, 1×1 convolution) ───────────────────────┘
```

1. **Lift:** v₀(x) = P(a(x), x). A pointwise linear layer maps the input values (and usually the grid coordinates, appended as extra channels) to `width` channels, for example 32 or 64.
2. **Fourier layers** (typically four):

```math
v_{l+1}(x) = \sigma\Big( W_l\, v_l(x) + \mathcal{F}^{-1}\big( R_l \cdot \mathcal{F}(v_l) \big)(x) \Big)
```

   - 𝓕 is the FFT over the spatial dimensions;
   - R_l holds learned complex weights for the lowest k modes in each dimension (a channel-mixing matrix per mode); all higher modes are set to zero;
   - W_l is a pointwise linear map (a 1 × 1 convolution), the local path;
   - σ is a nonlinearity such as GELU.
3. **Project:** u(x) = Q(v_L(x)), a pointwise map to the output channels.

## The Fourier layer in detail

**Why Fourier space?** The neural-operator layer contains an integral ∫κ(x, y)v(y)dy. If the kernel depends only on the difference, κ(x − y), the integral is a convolution, and by the convolution theorem a convolution is a multiplication in Fourier space: 𝓕(κ * v) = 𝓕(κ) · 𝓕(v). So FNO parametrizes the kernel directly by its Fourier coefficients R and applies it with two FFTs. See [The neural-operator layer](neural-operators.md#the-neural-operator-layer-a-learned-integral-kernel).

**What this buys:**

- **A global receptive field in one layer:** every output point depends on every input point, which suits elliptic problems and long-range interactions (pressure, for example), where a CNN would need many layers.
- **O(N log N) cost** per layer for N grid points.
- **Resolution-independent weights:** R lives on modes, not grid points, so the same weights apply to a 64 × 64 or a 256 × 256 grid (the lowest k modes exist at both).

**The two paths have different jobs.** The spectral path carries smooth, global information. The pointwise W path carries local and high-frequency information that the truncated spectral path drops, and it lets the layer represent non-periodic behaviour at the boundaries. Removing W hurts accuracy noticeably.

**Parameter count.** Per layer, the spectral weights have about k^d × width² complex numbers in d dimensions (for 2D real FFTs, two sets of k × k modes). With width 32 and k = 12 in 2D, that is about 12 × 12 × 2 × 32 × 32 ≈ 295,000 complex weights per layer, so the number of modes, not the grid size, drives model size, especially in 3D.

## Why FNO keeps only a few Fourier modes

1. **Smooth solutions live in low frequencies.** Solutions of diffusive and elliptic PDEs have Fourier coefficients that decay quickly, so a few modes carry almost all of the energy. The code below shows a smooth field reproduced exactly by four modes.
2. **Resolution independence.** Fixing k modes ties the learned kernel to physical frequencies, not to the grid. Any grid fine enough to resolve those k modes can be used.
3. **Fewer parameters and regularization.** Weights grow with k^d × width²; truncation keeps the model small and reduces overfitting, which matters with a few hundred to a few thousand training samples. It also filters noise.
4. **Higher modes may not be learnable anyway** from limited data.

**The cost of truncating:** the spectral path cannot represent fine-scale detail. Sharp fronts, boundary layers, shocks and discontinuities need many modes; truncated, they are smeared and show Gibbs-like ringing. The pointwise W path and the nonlinearities can reintroduce some high frequencies, but FNO is at its best on smooth solutions. For problems dominated by sharp features, consider more modes, multi-scale architectures (U-Net-like or CNO), or local methods.

## Code: what mode truncation does to smooth and sharp fields

Keep only the first k Fourier modes of three periodic fields (what the spectral path of a Fourier layer can represent) and measure the relative L2 error.

```python
import numpy as np

n = 1024
x = np.arange(n) / n
fields = {
    "smooth (two sine waves)":   np.sin(2 * np.pi * x) + 0.3 * np.sin(2 * np.pi * 3 * x),
    "steep front (width 0.01)":  np.tanh((x - 0.5) / 0.01) - np.tanh((x - 0.9) / 0.01),
    "discontinuity (a step)":    ((x > 0.5) & (x < 0.9)).astype(float),
}

def keep_modes(u, k):
    """Low-pass a periodic signal: keep only the first k Fourier modes, as an FNO layer does."""
    u_hat = np.fft.rfft(u)
    u_hat[k:] = 0
    return np.fft.irfft(u_hat, n=len(u))

rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
print("relative L2 error after keeping the first k modes")
print(f"{'':27s}" + "".join(f"k = {k:<6d}" for k in (4, 12, 32, 128)))
for name, u in fields.items():
    print(f"{name:27s}" + "".join(f"{rel(keep_modes(u, k), u):<10.4f}" for k in (4, 12, 32, 128)))
```

Output:

```
relative L2 error after keeping the first k modes
                           k = 4     k = 12    k = 32    k = 128
smooth (two sine waves)    0.0000    0.0000    0.0000    0.0000
steep front (width 0.01)   0.2400    0.0743    0.0098    0.0000
discontinuity (a step)     0.2807    0.1466    0.0892    0.0434
```

Smooth fields need very few modes; a steep front needs a few dozen; a true discontinuity converges slowly (roughly like k^(−1/2) in this norm) and rings. That is the trade-off FNO makes by truncating.

## Limitations, Geo-FNO and GINO

| Limitation | Why | Remedies |
| --- | --- | --- |
| **Uniform grids** | The FFT needs equally spaced points | Interpolate to a grid (loses boundary detail); Geo-FNO; GINO; or use graph or transformer operators |
| **Periodicity** | The discrete Fourier transform treats the domain as periodic | Zero-pad the domain; the W path helps; non-periodic spectral bases |
| **Complex geometry** | Bodies, holes and curved boundaries do not fit a box grid | Geo-FNO, GINO, masks with signed distance functions |
| **Memory in 3D** | Spectral weights scale with k³ × width² | Factorized FNO (separate transforms per dimension), fewer modes, smaller width |
| **Sharp features** | Mode truncation (above) | More modes, multi-scale or alias-free architectures (CNO) |
| **"Zero-shot super-resolution"** | Running on a finer grid is possible | It does not add physics that the training data never resolved; test it |

**Geo-FNO** (Li et al., 2022) learns a **deformation** from the irregular physical domain to a regular latent domain (or takes one as input), applies FNO on the latent uniform grid, and maps back. It handles, for example, airfoils, pipes and elasticity problems on irregular meshes, as long as the geometry can be smoothly mapped to a simple latent domain.

**GINO** (Li et al., 2023) handles large 3D geometries: a graph neural operator encoder maps the input point cloud and a signed distance function onto a latent regular grid, FNO processes it there, and a graph neural operator decoder maps back to the surface or query points. It was demonstrated on 3D car aerodynamics with Reynolds numbers up to five million, predicting surface pressure from about five hundred training examples. See [Graph neural operators](neural-operators.md#graph-neural-operators).

## Variants

- **Factorized FNO (F-FNO):** Fourier transforms factorized over each dimension separately, with fewer parameters, allowing deeper models and improving accuracy.
- **U-NO:** a U-shaped stack of operator layers with varying resolutions.
- **AFNO (adaptive Fourier neural operator):** used as token mixing in FourCastNet for global weather forecasting.
- **Spherical FNO:** spherical harmonics instead of Fourier modes, for data on the sphere (weather and climate).
- **Physics-informed FNO (PINO):** adds PDE residuals to the loss. See [Physics-informed neural operator](physics-informed-neural-operator.md).

## Code: a 1D Fourier layer

```python
import torch, torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, c_in, c_out, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.randn(c_in, c_out, modes, dtype=torch.cfloat))

    def forward(self, x):                        # x: (batch, c_in, n)
        x_ft = torch.fft.rfft(x)                 # (batch, c_in, n // 2 + 1)
        out_ft = torch.zeros(x.shape[0], self.weight.shape[1], x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.shape[-1])      # keep only the lowest modes
        out_ft[..., :m] = torch.einsum("bik,iok->bok", x_ft[..., :m], self.weight[..., :m])
        return torch.fft.irfft(out_ft, n=x.shape[-1])

class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)   # the "W" path

    def forward(self, x):
        return nn.functional.gelu(self.spectral(x) + self.pointwise(x))

block = FNOBlock1d(width=16, modes=12)
print(block(torch.randn(4, 16, 64)).shape, block(torch.randn(4, 16, 256)).shape)  # same weights, two grids
```

Output:

```
torch.Size([4, 16, 64]) torch.Size([4, 16, 256])
```

The same weights run on a 64-point and a 256-point grid. A full 1D FNO adds a lifting layer (input values plus the coordinate, to `width` channels), four such blocks and a projection layer.

## In an interview

**Thirty-second answer to "Explain the Fourier Neural Operator."** "Lift the input function to many channels pointwise, then apply Fourier layers: FFT, multiply the lowest k modes by learned complex weights per channel pair, zero the rest, inverse FFT, add a pointwise linear path and a nonlinearity. That is a learned global convolution, O(N log N), with weights tied to modes rather than grid points, so it runs on different resolutions. Finally project to the output channels. It is fast and accurate on regular grids, but needs uniform grids and assumes periodicity, so irregular geometry uses Geo-FNO or GINO."

**Likely follow-ups**

- *Why keep only a few modes?* Smooth solutions live in low frequencies; it makes the kernel resolution-independent, cuts parameters (k^d × width²) and regularizes; but sharp features suffer.
- *FNO vs U-Net on a grid?* A well-tuned U-Net is a strong baseline; FNO's global receptive field helps for elliptic, long-range coupling. Compare on your data.
- *What does the W path do?* Carries local, high-frequency and non-periodic information the truncated spectral path misses.

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al.)](https://arxiv.org/abs/2010.08895) and [original code](https://github.com/zongyi-li/fourier_neural_operator)
- [Zongyi Li's blog post on the Fourier Neural Operator](https://zongyi-li.github.io/blog/2020/fourier-pde/)
- [Fourier Neural Operator with Learned Deformations for PDEs on General Geometries (Geo-FNO)](https://arxiv.org/abs/2207.05209)
- [Geometry-Informed Neural Operator for Large-Scale 3D PDEs (GINO)](https://arxiv.org/abs/2309.00583)
- [Factorized Fourier Neural Operators](https://arxiv.org/abs/2111.13802)
- [FourCastNet (AFNO for weather)](https://arxiv.org/abs/2202.11214)
- [ETH Zürich AISE: Fourier Neural Operators (video)](https://www.youtube.com/watch?v=b96wRdjH1Lg)
- [neuraloperator library](https://github.com/neuraloperator/neuraloperator)
