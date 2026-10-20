# DeepONet

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** DeepONet (Lu, Jin and Karniadakis, 2021) learns an operator G that maps an input function u to an output function G(u), evaluated at any query point y. It has two networks. The **branch network** takes the input function sampled at m fixed *sensor* locations and outputs p coefficients b_k(u). The **trunk network** takes the query coordinate y and outputs p basis functions t_k(y). The prediction is their dot product, G(u)(y) ≈ Σ_k b_k(u) t_k(y) + b₀: a learned basis in the output variable, with coefficients that depend on the input function. The design comes directly from Chen and Chen's universal approximation theorem for operators. Compared with the Fourier Neural Operator, DeepONet is more flexible about **where outputs are evaluated** (any point, any geometry) but needs inputs at **fixed sensors**, and its linear reconstruction struggles with moving features such as shocks. In a large comparison by Lu et al., the two performed comparably on simpler problems, while FNO's accuracy deteriorated much more on complex geometries and especially noisy data.

**On this page**

- [Branch and trunk networks](#branch-and-trunk-networks)
- [Why the dot product works](#why-the-dot-product-works)
- [Code: a DeepONet that learns integration](#code-a-deeponet-that-learns-integration)
- [Variants](#variants)
- [Limitations](#limitations)
- [DeepONet vs FNO](#deeponet-vs-fno)
- [In an interview](#in-an-interview)

## Branch and trunk networks

```
input function u, sampled at m fixed sensors
u(x₁), …, u(x_m) ──▶ branch network ──▶ b₁(u), …, b_p(u)  ─┐
                                                           ├─▶  Σ_k b_k(u) · t_k(y) + b₀  =  G(u)(y)
query point y ──────▶ trunk network  ──▶ t₁(y), …, t_p(y)  ─┘
```

- **The branch network** encodes the input function. It sees the function only through its values at the m sensor points, which must be the *same* locations for every training and test sample. Any network that maps a vector of m values to p outputs works: an MLP, a CNN if the sensors form a grid, or a set or graph encoder if they do not.
- **The trunk network** encodes the output location. It takes a coordinate y (in space, or space and time) and outputs p values. As a function of y, each output t_k(y) is a learned basis function.
- **The combination** is a dot product plus a bias. For a vector-valued output, use separate groups of basis functions per output component, or split the p outputs among them.

**Example:** for heat conduction with a variable source, u is the source term sampled at 100 sensors in the domain, y is a point where the temperature is wanted, and G(u)(y) is the temperature there.

## Why the dot product works

**The math.** Chen and Chen (1995) proved that a network of exactly this form, a sum of products of a function of the input-function samples and a function of the query location, can approximate any continuous nonlinear operator on a compact set to arbitrary accuracy. Lu et al. turned the theorem's shallow construction into deep branch and trunk networks. See [The universal approximation theorem](universal-approximation.md#versions-for-deep-networks-and-for-operators).

**The intuition.** Many solution operators have outputs that live near a low-dimensional space of functions. The trunk network learns a basis for that space, and the branch network learns how much of each basis function a given input needs. This is exactly how reduced-order models work with POD modes, except that both the basis and the coefficient map are learned and nonlinear in the input. See [Reduced-order models](reduced-order-models.md).

**Cost advantage.** For a fixed input u, the branch is evaluated once; predicting at a thousand query points needs only the cheap trunk evaluations and dot products.

## Code: a DeepONet that learns integration

The classic first example from the DeepONet paper: learn the antiderivative operator G(u)(y) = ∫₀^y u(s) ds from random smooth input functions, then test it on functions it has never seen, at query points between the sensors.

```python
import math
import torch, torch.nn as nn

torch.manual_seed(0)
m, p, n_terms = 50, 64, 6                                  # sensors, basis functions, Fourier terms per input
sensors = torch.linspace(0, 1, m)
k = torch.arange(1, n_terms + 1).float()

def sample_coefficients(n):
    """Random smooth inputs u(x) = sum_k a_k sin(k pi x) + b_k cos(k pi x), amplitudes decaying like 1/k."""
    return torch.randn(n, n_terms) / k, torch.randn(n, n_terms) / k

def u_at(a, b, x):                                         # input function values, shape (n, len(x))
    return a @ torch.sin(math.pi * k[:, None] * x) + b @ torch.cos(math.pi * k[:, None] * x)

def G_at(a, b, y):                                         # the operator to learn: G(u)(y) = integral_0^y u(s) ds
    return (a / (math.pi * k)) @ (1 - torch.cos(math.pi * k[:, None] * y)) + (b / (math.pi * k)) @ torch.sin(math.pi * k[:, None] * y)

class DeepONet(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(m, width), nn.Tanh(), nn.Linear(width, p))       # u at sensors -> b_k
        self.trunk = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, p))                                         # y -> t_k(y)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, u_sensors, y):               # u_sensors: (batch, m), y: (n_query,)
        return self.branch(u_sensors) @ self.trunk(y[:, None]).T + self.bias    # sum_k b_k(u) t_k(y) + b_0

a_tr, b_tr = sample_coefficients(2000)
u_tr = u_at(a_tr, b_tr, sensors)                          # the only thing the model sees about each input
model = DeepONet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
steps = 6000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
for step in range(steps):
    idx = torch.randint(0, 2000, (256,))                  # a mini-batch of input functions
    y = torch.rand(64)                                    # new query points every step
    loss = ((model(u_tr[idx], y) - G_at(a_tr[idx], b_tr[idx], y)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

a_te, b_te = sample_coefficients(200)                     # 200 input functions never seen in training
y_test = torch.linspace(0, 1, 201)                        # query anywhere, including between sensors
with torch.no_grad():
    pred, true = model(u_at(a_te, b_te, sensors), y_test), G_at(a_te, b_te, y_test)
rel_error = ((pred - true).norm() / true.norm()).item()
print(f"antiderivative operator, 200 unseen input functions: relative L2 error {rel_error:.4f}")
```

Output:

```
antiderivative operator, 200 unseen input functions: relative L2 error 0.0313
```

About 3% error on unseen input functions, evaluated on a 201-point grid although the inputs were only ever given at 50 sensors. Longer training, more input functions and larger networks reduce it further.

## Variants

- **POD-DeepONet:** replaces the trunk network with POD modes computed from the training outputs; the branch predicts the mode coefficients. Strong when outputs share a fixed grid.
- **Physics-informed DeepONet** (Wang, Wang and Perdikaris, 2021): adds the PDE residual, computed by automatic differentiation through the trunk's coordinate inputs, to the loss. It improves accuracy, reduces the need for paired data, and can even solve parametric PDEs without any input-output pairs. See [Physics-informed neural operator](physics-informed-neural-operator.md#related-approaches).
- **Multiple-input operators (MIONet):** one branch network per input function (for example a source and a boundary condition), combined with the trunk.
- **Practical extensions** from Lu et al.'s comparison study: feature expansions, normalization and architectural choices that give the model better inductive biases, speed up training and bring its cost closer to FNO's.
- **Geometry-aware branches:** set, graph or point-cloud encoders in the branch so that inputs need not share the same sensors across samples.

## Limitations

1. **Fixed sensors for the input.** The branch network needs the input function at the *same* m locations in every sample. If geometries or meshes differ between samples, inputs must be interpolated to common sensors, or the branch replaced by a set or graph encoder.
2. **Linear reconstruction in the output.** For a given input, the output is a linear combination of p fixed basis functions t_k(y). Problems whose features **move** with the input, such as advected fronts, shocks and travelling waves, need very many basis functions (the same limitation as linear reduced-order models). Nonlinear decoders and shifted or transported bases address it.
3. **Basis size p is a hyperparameter** that trades accuracy against overfitting.
4. **Data hunger:** like other operators, a pure data-driven DeepONet needs many solved instances; physics-informed training reduces this.

## DeepONet vs FNO

The comparison by Lu et al. (2022) evaluated both, with practical extensions to each, on 16 benchmarks using FAIR data. Their main findings: in **relatively simple settings the two perform comparably**, but **for complex geometries and especially noisy data, the performance of FNO deteriorates greatly**, while DeepONet was more robust.

| | DeepONet | FNO |
| --- | --- | --- |
| Output locations | Any query point (trunk takes coordinates) | Grid points (Geo-FNO and GINO for irregular geometry) |
| Input discretization | Fixed sensor locations | Regular grid; any resolution |
| Inductive bias | Learned basis × learned coefficients | Global convolution in Fourier space |
| Strengths | Flexible output queries, scattered sensors, complex output domains, physics-informed training via autodiff | Speed and accuracy on regular grids; global receptive field; resolution flexibility |
| Weaknesses | Fixed input sensors; moving features need many basis functions | Uniform grids and periodicity; sensitivity to noise and complex geometry |

**How to choose:** if the data is on a regular grid and smooth, try FNO first. If outputs are needed at arbitrary points, inputs come from fixed sensors (for example measurements), or the output domain is irregular, try DeepONet. In both cases, compare against a simple baseline and on your own data.

## In an interview

**Thirty-second answer to "Explain DeepONet."** "Two networks: the branch takes the input function sampled at fixed sensors and outputs p coefficients; the trunk takes a query coordinate and outputs p basis-function values. The prediction is their dot product plus a bias, so the trunk learns a basis for the solutions and the branch learns how much of each basis function an input needs. It is grounded in Chen and Chen's universal approximation theorem for operators, can be queried anywhere, but needs fixed input sensors and struggles with moving features because the reconstruction is linear in the basis."

**Likely follow-ups**

- *DeepONet vs FNO?* Comparable on simple problems; FNO is fast on regular grids; DeepONet is more flexible in output locations and was more robust on complex geometries and noisy data in Lu et al.'s comparison.
- *How do you handle inputs on different meshes?* Interpolate to fixed sensors, or replace the branch with a set or graph encoder.
- *How would you add physics?* Physics-informed DeepONet: differentiate the output with respect to the trunk input with autograd and add the PDE residual to the loss.

## References

- [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators (Lu, Jin & Karniadakis)](https://arxiv.org/abs/1910.03193) and [code](https://github.com/lululxvi/deeponet)
- [A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data (Lu et al.)](https://arxiv.org/abs/2111.05512)
- [Learning the solution operator of parametric PDEs with physics-informed DeepONets (Wang, Wang & Perdikaris)](https://arxiv.org/abs/2103.10974)
- T. Chen and H. Chen, "Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems", *IEEE Transactions on Neural Networks*, 1995.
- [ETH Zürich AISE: Spectral Neural Operators and Deep Operator Networks (video)](https://www.youtube.com/watch?v=BxklDO0TMlA)
- [DeepXDE (includes DeepONet)](https://deepxde.readthedocs.io/)
