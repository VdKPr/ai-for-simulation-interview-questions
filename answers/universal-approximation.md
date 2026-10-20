# The Universal Approximation Theorem

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** The universal approximation theorem says that a feed-forward neural network with a single hidden layer and a suitable non-linear activation can approximate any continuous function on a bounded, closed domain to any desired accuracy, provided the hidden layer is wide enough. Versions exist for deep narrow networks and for *operators* (maps between functions), which is the theoretical starting point of DeepONet and neural operators. What it does **not** say is just as important: it does not say how many neurons are needed (possibly astronomically many), that gradient descent will find the right weights, that a network trained on finite, noisy data will generalize, that it will behave sensibly outside the domain it was trained on, or that its derivatives will be accurate. It is an existence result about *expressiveness*, not a guarantee about *learning*.

**On this page**

- [What the theorem says](#what-the-theorem-says)
- [Versions for deep networks and for operators](#versions-for-deep-networks-and-for-operators)
- [What it does not guarantee](#what-it-does-not-guarantee)
- [Code: more width helps inside the domain, nothing helps outside it](#code-more-width-helps-inside-the-domain-nothing-helps-outside-it)
- [Approximation, estimation and optimization error](#approximation-estimation-and-optimization-error)
- [In an interview](#in-an-interview)

## What the theorem says

**Informal statement.** Let K be a compact (closed and bounded) subset of ℝⁿ and f a continuous function on K. For any tolerance ε > 0, there is a one-hidden-layer network

```math
g(\mathbf{x}) = \sum_{j=1}^{N} a_j\, \sigma(\mathbf{w}_j \cdot \mathbf{x} + b_j)
```

with finitely many neurons N such that |g(x) − f(x)| < ε for every x in K.

**History in one line each:**

- Cybenko (1989) proved it for sigmoid activations;
- Hornik, Stinchcombe and White (1989) for general squashing functions;
- Leshno, Lin, Pinkus and Schocken (1993) showed that it holds exactly when the activation is **not a polynomial** (ReLU, tanh, sigmoid all qualify; a polynomial activation does not, because sums of polynomials of bounded degree stay polynomials of that degree).

**Intuition.** Each hidden neuron makes a "ramp" or "step" along some direction. Enough of them, shifted and scaled, can build narrow bumps, and enough bumps can tile any continuous function, like approximating a curve with a histogram.

## Versions for deep networks and for operators

- **Deep, narrow networks.** Universality also holds when width is bounded and depth grows: for example, ReLU networks of width n + 4 (with n inputs) are universal (Lu et al., 2017). Depth can be exponentially more efficient than width for some function classes, which is part of why deep networks are used.
- **Operators.** Chen and Chen (1995) proved that a network of a particular two-part form can approximate any continuous nonlinear *operator*, a map from functions to functions, such as "boundary condition → solution". DeepONet's branch-and-trunk structure is built directly on this result. See [DeepONet](deeponet.md).
- **Neural operators.** Kovachki et al. proved universal approximation for the neural-operator framework, including the Fourier neural operator; similar results exist for convolutional neural operators. See [Neural operators](neural-operators.md).
- **Derivatives.** With smooth activations, networks can also approximate a function *and* its derivatives (in Sobolev norms). This matters for PINNs, whose loss contains derivatives of the network. ReLU networks have zero second derivatives almost everywhere, so they cannot represent a PDE residual with second derivatives. See [Higher derivatives](automatic-differentiation.md#higher-derivatives-as-pinns-need).

## What it does not guarantee

1. **Size.** The number of neurons needed can grow exponentially with the input dimension for general continuous functions (the curse of dimensionality). Efficient approximation needs structure: smoothness, compositional structure, symmetries.
2. **Trainability.** The theorem says good weights *exist*, not that gradient descent will find them. Optimization can stall, especially for physics losses. See [PINN training failures](pinn-training-failures.md).
3. **Generalization from finite data.** It says nothing about learning from 200 noisy simulations. A network that can represent the right function can equally represent many wrong functions that fit the training data.
4. **Extrapolation.** The guarantee holds only on the compact set K. Outside it, the network can do anything, and usually does something unhelpful. The code below shows this.
5. **Rates and efficiency.** How fast the error falls with network size depends on the smoothness of the target; for rough targets (shocks, singularities) it can be slow. See [Spectral bias](spectral-bias-and-fourier-features.md#spectral-bias).
6. **Physics.** Nothing ensures conservation, symmetry, stability or positivity. Those must come from the architecture, the loss or the data.
7. **That neural networks are the best choice.** Polynomials, splines, Fourier series and Gaussian processes with suitable kernels are universal approximators too. Universality is not what distinguishes networks; scalability and inductive biases are.

## Code: more width helps inside the domain, nothing helps outside it

Fit sin(x) on [−π, π] with one-hidden-layer tanh networks of increasing width, then evaluate them on [π, 2π], outside the training range.

```python
import math
import torch

torch.manual_seed(0)
x_train = torch.linspace(-math.pi, math.pi, 400).unsqueeze(1)
x_out = torch.linspace(math.pi, 2 * math.pi, 200).unsqueeze(1)            # outside the training interval
f = torch.sin

errors = {}
for width in [2, 8, 64]:
    net = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for step in range(4000):
        loss = ((net(x_train) - f(x_train)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        err_in = (net(x_train) - f(x_train)).abs().max().item()
        err_out = (net(x_out) - f(x_out)).abs().max().item()
    errors[width] = (err_in, err_out)
    print(f"width {width:3d}: max error inside [-pi, pi] = {err_in:.4f}, outside [pi, 2pi] = {err_out:.3f}")
```

Output:

```
width   2: max error inside [-pi, pi] = 0.1145, outside [pi, 2pi] = 1.928
width   8: max error inside [-pi, pi] = 0.0039, outside [pi, 2pi] = 1.171
width  64: max error inside [-pi, pi] = 0.0025, outside [pi, 2pi] = 1.976
```

Inside the training interval, more width gives a better approximation, as the theorem promises. Outside, the error is of the same order as the function itself for every width: the theorem has nothing to say there, and neither does the network. The same is true of a surrogate asked about designs outside its training distribution. See [Interpolation vs extrapolation](out-of-distribution-and-generalization.md#interpolation-vs-extrapolation).

## Approximation, estimation and optimization error

A useful way to explain where a trained model's error comes from:

```
total error  =  approximation error   (can the model class represent the target at all?)
             +  estimation error      (finite, noisy data: does the fit generalize?)
             +  optimization error    (did training find good weights?)
```

The universal approximation theorem addresses only the first term, and only in the limit of large networks. In practice, for surrogates trained on a few hundred simulations, estimation error usually dominates; for PINNs, optimization error often dominates. Improvements come from attacking the dominant term: more or better data and inductive biases for estimation, better training recipes for optimization.

## In an interview

**Thirty-second answer.** "It says a network with one hidden layer and a non-polynomial activation can approximate any continuous function on a compact set as closely as you like, if it is wide enough; there are versions for deep narrow networks and for operators, which underpin DeepONet and neural operators. It does not say how many neurons you need, that training will find the weights, that the model will generalize from finite data, or anything about behaviour outside the training domain. It is about expressiveness, not learnability."

**Likely follow-ups**

- *Does it justify using a network for my surrogate?* Only that a good approximation exists; data, inductive biases and training decide whether you find it.
- *Why can't ReLU networks be used for PINNs with second derivatives?* Their second derivative is zero almost everywhere; use smooth activations.
- *Which theorem is behind DeepONet?* Chen and Chen's 1995 universal approximation theorem for nonlinear operators.

## References

- G. Cybenko, "Approximation by superpositions of a sigmoidal function", *Mathematics of Control, Signals and Systems*, 1989.
- K. Hornik, M. Stinchcombe and H. White, "Multilayer feedforward networks are universal approximators", *Neural Networks*, 1989.
- M. Leshno, V. Ya. Lin, A. Pinkus and S. Schocken, "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function", *Neural Networks*, 1993.
- T. Chen and H. Chen, "Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems", *IEEE Transactions on Neural Networks*, 1995.
- [The Expressive Power of Neural Networks: A View from the Width (Lu et al., 2017)](https://arxiv.org/abs/1709.02540)
- [DeepONet (Lu, Jin & Karniadakis)](https://arxiv.org/abs/1910.03193)
- [Neural Operator: Learning Maps Between Function Spaces (Kovachki et al.)](https://arxiv.org/abs/2108.08481)
