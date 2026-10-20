# Automatic Differentiation

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Automatic differentiation (AD) computes exact derivatives of a program by applying the chain rule to every elementary operation it executes (add, multiply, sin, exp, matrix multiply). It is not **finite differences**, which approximate a derivative from function values at nearby points and suffer from truncation error (step too large) and round-off error (step too small). It is not **symbolic differentiation**, which manipulates formulas and can produce huge expressions. **Forward mode** carries derivatives along with values and costs about one extra pass per *input* direction; **reverse mode** (backpropagation) records the computation and sweeps backwards, giving the gradient with respect to *all* inputs for about the cost of one extra pass per *output*. Reverse mode is why neural networks with millions of parameters can be trained, why PINNs can compute PDE residuals, and it is closely related to the adjoint method used for gradients in simulation.

**On this page**

- [Three ways to get a derivative](#three-ways-to-get-a-derivative)
- [Finite differences and their two errors](#finite-differences-and-their-two-errors)
- [Forward mode](#forward-mode)
- [Reverse mode](#reverse-mode)
- [Jacobian-vector and vector-Jacobian products](#jacobian-vector-and-vector-jacobian-products)
- [Higher derivatives, as PINNs need](#higher-derivatives-as-pinns-need)
- [Differentiating through a simulation](#differentiating-through-a-simulation)
- [Code: dual numbers, finite differences and PyTorch](#code-dual-numbers-finite-differences-and-pytorch)
- [In an interview](#in-an-interview)

## Three ways to get a derivative

Take f(x) = sin(x²)·eˣ + 3x and ask for f′(1.3).

| Method | How | Accuracy | Cost | Problem |
| --- | --- | --- | --- | --- |
| **Symbolic** | Differentiate the formula: f′ = (2x cos x² + sin x²) eˣ + 3 | Exact | Cheap here | Expressions can grow enormously ("expression swell"); needs a formula, not a program with loops and branches |
| **Finite differences** | (f(x + h) − f(x)) / h | Approximate | One extra evaluation per input | Truncation vs round-off trade-off; n inputs need n extra evaluations |
| **Automatic differentiation** | Apply the chain rule to each operation the program performs | Exact to floating-point precision | A small constant times one evaluation | Memory (reverse mode); code must use differentiable operations |

AD works on the *program*, including loops, branches and function calls, not on a closed-form formula. It does not build a big symbolic expression: it propagates numbers.

## Finite differences and their two errors

The forward difference (f(x + h) − f(x))/h has two error sources:

- **Truncation error** from the Taylor series: about (h/2)|f″(x)|, so it shrinks as h shrinks.
- **Round-off error**: f(x + h) and f(x) are nearly equal, and their difference loses significant digits. With machine precision ε ≈ 10⁻¹⁶ in double precision, this error is about ε|f(x)|/h, so it *grows* as h shrinks.

The total error is smallest around h ≈ √ε ≈ 10⁻⁸ (for a well-scaled function), where it is about 10⁻⁸ relative: only half the digits of the arithmetic. The code below shows exactly this: errors of 1.4 at h = 10⁻¹, 1.2 × 10⁻³ at 10⁻⁴, 1.7 × 10⁻⁷ at 10⁻⁸, and back up to 2.4 × 10⁻⁴ at 10⁻¹².

**Central differences** (f(x + h) − f(x − h))/(2h) have truncation error proportional to h² and a best accuracy around 10⁻¹¹, at twice the cost. **Complex-step differentiation**, Im(f(x + ih))/h, has no subtraction and therefore no round-off problem, for functions that can be evaluated in complex arithmetic.

Finite differences remain useful for one thing: **checking** an AD or adjoint gradient on a few random directions ("gradient checking").

## Forward mode

**Idea:** carry the derivative alongside the value through the computation. Every number becomes a pair (value, derivative), sometimes called a *dual number*, and every operation updates both:

```
(a, a′) + (b, b′) = (a + b,  a′ + b′)
(a, a′) × (b, b′) = (a·b,    a′·b + a·b′)        product rule
sin((a, a′))      = (sin a,  cos a · a′)          chain rule
```

Seed the input with derivative 1 and read the derivative of the output at the end. For f(x) = sin(x²)·eˣ + 3x at x = 1.3, starting from (1.3, 1):

```
x²         = (1.69,   2·1.3·1 = 2.6)
sin(x²)    = (0.9929, cos(1.69)·2.6 = −0.3092)
eˣ         = (3.6693, 3.6693)
product    = (3.6433, −0.3092·3.6693 + 0.9929·3.6693 = 2.5087)
+ 3x       = (7.5433, 2.5087 + 3 = 5.5087)       f′(1.3) ≈ 5.5087
```

**Cost:** one forward pass gives the derivative with respect to **one input direction**. For a function with n inputs and m outputs, the full Jacobian needs n passes. Forward mode is efficient when there are **few inputs and many outputs**.

## Reverse mode

**Idea:** run the program forward once, recording each operation and its intermediate values (the "tape" or computational graph). Then go **backwards** from the output, propagating the *adjoint* of each intermediate variable, ∂output/∂(intermediate), using the chain rule in reverse order.

For a scalar loss L of a neural network with millions of parameters θ, one forward pass and one backward pass give **all** of ∂L/∂θ, for a small constant times the cost of the forward pass. That is backpropagation. See [Backpropagation and logistic regression](backpropagation-and-logistic-regression.md).

**Cost:** one backward pass per **output**. Reverse mode is efficient when there are **many inputs and few outputs**, which is exactly the situation of a loss function (one output, millions of parameters) and of design optimization (one objective such as drag, thousands of shape parameters).

**The price is memory:** the intermediate values of the forward pass must be stored for the backward pass. For deep networks on large meshes this dominates GPU memory, which is why **gradient checkpointing** exists: store only some intermediate values and recompute the others during the backward pass. See [Gradient checkpointing](scaling-to-large-meshes.md#gradient-checkpointing).

## Jacobian-vector and vector-Jacobian products

For a function F: ℝⁿ → ℝᵐ with Jacobian J (m × n):

- **Forward mode** computes a **Jacobian-vector product (JVP)**, J·v, for about the cost of one evaluation of F. It answers: "how does the output change if I move the input in direction v?"
- **Reverse mode** computes a **vector-Jacobian product (VJP)**, uᵀ·J, for about the cost of one evaluation plus storage. It answers: "how does a weighted combination of outputs depend on every input?"

Neither forms the full Jacobian unless asked to. This matters in simulation: Newton-Krylov solvers only need J·v products, and adjoint-based optimization only needs uᵀJ. In PyTorch, `torch.func.jvp` and `torch.func.vjp` expose both; JAX has `jax.jvp` and `jax.vjp`.

## Higher derivatives, as PINNs need

A PINN for the heat equation needs ∂u/∂t and ∂²u/∂x² of the network output u(x, t) with respect to its *inputs*. AD gets them by differentiating the derivative computation again (**nested AD**). In PyTorch this means calling `torch.autograd.grad(..., create_graph=True)` so that the first derivative is itself recorded and can be differentiated, as in the code below. See [Derivatives in the PDE residual](pinns.md#derivatives-in-the-pde-residual).

Two practical consequences:

- **Cost grows with derivative order:** each nesting adds another backward pass through a larger graph. A fourth-order PDE (plates, beams) is noticeably more expensive.
- **The network must be smooth enough:** with ReLU activations, the second derivative is zero almost everywhere, so a ReLU network cannot represent u_xx. PINNs use smooth activations such as tanh, sine or GELU-like functions.

## Differentiating through a simulation

To optimize a design with gradients, you need d(objective)/d(design parameters) where the objective depends on a simulation. Options:

1. **AD through every solver step** (unrolling): conceptually simple in a differentiable framework (JAX, PyTorch, Warp), but stores every iteration and every time step.
2. **Implicit differentiation / the adjoint method:** for a converged solution of R(u, p) = 0, differentiate the *equation* rather than the iterations. One extra linear solve with the transposed Jacobian gives the gradient with respect to all parameters at once. This is reverse-mode AD applied at the level of the equation. See [The adjoint method](adjoint-method.md).
3. **AD through a trained surrogate:** cheap and easy, but the gradient is only as good as the surrogate, and optimizers can exploit its errors. See [Surrogate gradients vs the adjoint](surrogate-based-optimization.md#surrogate-gradients-vs-the-adjoint).

## Code: dual numbers, finite differences and PyTorch

```python
import math
import torch

class Dual:
    """Forward-mode AD: carry (value, derivative) through every operation."""
    def __init__(self, val, der=0.0):
        self.val, self.der = val, der
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val + o.val, self.der + o.der)
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val * o.val, self.der * o.val + self.val * o.der)   # product rule
    __rmul__ = __mul__

def sin(x):
    return Dual(math.sin(x.val), math.cos(x.val) * x.der)                   # chain rule

def exp(x):
    return Dual(math.exp(x.val), math.exp(x.val) * x.der)

def f(x):
    return sin(x * x) * exp(x) + 3 * x                                       # f(x) = sin(x^2) e^x + 3x

def f_float(x):
    return math.sin(x * x) * math.exp(x) + 3 * x

x0 = 1.3
exact = (2 * x0 * math.cos(x0**2) + math.sin(x0**2)) * math.exp(x0) + 3    # derivative by hand
ad_value = f(Dual(x0, 1.0)).der
print(f"forward-mode AD : {ad_value:.15f}")
print(f"by hand         : {exact:.15f}")

for h in [1e-1, 1e-4, 1e-8, 1e-12]:                                          # finite differences
    fd = (f_float(x0 + h) - f_float(x0)) / h
    print(f"forward FD h={h:.0e}: error {abs(fd - exact):.1e}")

xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)              # reverse mode (PyTorch)
y = torch.sin(xt**2) * torch.exp(xt) + 3 * xt
(g,) = torch.autograd.grad(y, xt, create_graph=True)
(g2,) = torch.autograd.grad(g, xt)                                         # second derivative: AD of AD
print(f"reverse-mode AD : {g.item():.15f}, second derivative {g2.item():.6f}")
```

Output:

```
forward-mode AD : 5.508725911613541
by hand         : 5.508725911613541
forward FD h=1e-01: error 1.4e+00
forward FD h=1e-04: error 1.2e-03
forward FD h=1e-08: error 1.7e-07
forward FD h=1e-12: error 2.4e-04
reverse-mode AD : 5.508725911613541, second derivative -24.126948
```

Both AD modes agree with the hand derivative to all printed digits. Finite differences are best near h = 10⁻⁸ and get worse in both directions.

## In an interview

**Thirty-second answer.** "AD applies the chain rule to each operation the program executes, so it gives exact derivatives to machine precision. Finite differences approximate the derivative from nearby function values and trade truncation error against round-off, so the best you get is about half the digits; they also need one extra evaluation per input. Symbolic differentiation works on formulas and can blow up in size. Forward-mode AD costs a pass per input direction, reverse mode a pass per output, which is why reverse mode, backpropagation, gives the gradient of a scalar loss with respect to millions of parameters cheaply."

**Likely follow-ups**

- *Forward or reverse mode for a PINN residual?* Reverse mode (autograd) is the usual tool; for derivatives with respect to a few input coordinates at many points, forward mode (JVPs) can be faster, and frameworks like JAX make it easy to combine them.
- *Why does reverse mode use so much memory?* It stores intermediate values from the forward pass; checkpointing trades recomputation for memory.
- *How is AD related to the adjoint method?* The adjoint method is reverse-mode differentiation applied to the converged equations rather than to every solver iteration.

## References

- [Automatic differentiation in machine learning: a survey (Baydin, Pearlmutter, Radul & Siskind)](https://arxiv.org/abs/1502.05767)
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX: the autodiff cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd edition, SIAM, 2008.
