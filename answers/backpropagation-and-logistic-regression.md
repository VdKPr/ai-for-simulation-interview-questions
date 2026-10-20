# Backpropagation and Logistic Regression from Scratch

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** **Backpropagation** computes the gradient of a scalar loss with respect to every parameter of a network by applying the chain rule backwards through the computation: run the forward pass and keep the intermediate values, then, starting from ∂L/∂(output), multiply by the local derivative of each operation in reverse order, accumulating gradients for weights and biases on the way. It costs roughly the same as one or two forward passes, whatever the number of parameters, which is what makes training large networks possible; it is reverse-mode automatic differentiation. **Logistic regression** is the simplest example: a linear score z = w·x + b passed through a sigmoid gives a probability p, the loss is binary cross-entropy, and the gradient has the clean form ∂L/∂w = (1/N) Σ (pᵢ − yᵢ) xᵢ. Implementing it from scratch, checking the gradient against finite differences and comparing with a library is a common coding question.

**On this page**

- [Logistic regression in one page](#logistic-regression-in-one-page)
- [Deriving the gradient](#deriving-the-gradient)
- [Code: logistic regression from scratch, with a gradient check](#code-logistic-regression-from-scratch-with-a-gradient-check)
- [Backpropagation in a neural network](#backpropagation-in-a-neural-network)
- [Code: a backward pass by hand, checked against autograd](#code-a-backward-pass-by-hand-checked-against-autograd)
- [Practical details interviewers ask about](#practical-details-interviewers-ask-about)
- [In an interview](#in-an-interview)

## Logistic regression in one page

**Model.** For an input vector x (for example normalized design features), a linear score passed through the sigmoid gives the probability of the positive class (for example "the design passes the requirement"):

```math
z = \mathbf{w}\cdot\mathbf{x} + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}
```

**Loss.** For labels yᵢ ∈ {0, 1}, the binary cross-entropy (the negative log-likelihood of a Bernoulli model), averaged over N samples, usually with an L2 penalty:

```math
L(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big] + \frac{\lambda}{2}\lVert\mathbf{w}\rVert^2
```

**Properties worth knowing:**

- The loss is **convex** in (w, b), so gradient descent finds the global minimum.
- Without regularization, perfectly separable data drives the weights to infinity; the L2 penalty prevents that.
- The decision boundary w·x + b = 0 is a hyperplane; logistic regression is a linear classifier.
- The output is a calibrated-looking probability, but calibration still has to be checked. See [Checking calibration](uncertainty-quantification.md#checking-calibration).

## Deriving the gradient

Use the chain rule through z. Two facts:

1. The sigmoid's derivative is σ′(z) = σ(z)(1 − σ(z)) = p(1 − p).
2. For one sample, the derivative of the cross-entropy with respect to p is −y/p + (1 − y)/(1 − p).

Multiply them:

```math
\frac{\partial L_i}{\partial z_i} = \left(-\frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i}\right) p_i(1 - p_i) = p_i - y_i
```

The sigmoid and the log cancel, which is why this pair is used together (and why the same simplification appears for softmax with cross-entropy). Since ∂z/∂w = x and ∂z/∂b = 1:

```math
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{N}\sum_i (p_i - y_i)\,\mathbf{x}_i + \lambda\mathbf{w}, \qquad \frac{\partial L}{\partial b} = \frac{1}{N}\sum_i (p_i - y_i)
```

In matrix form: ∇_w L = Xᵀ(p − y)/N + λw. The "error" p − y flows back and is multiplied by the input that produced it: the essence of backpropagation in one layer.

## Code: logistic regression from scratch, with a gradient check

```python
import numpy as np

rng = np.random.default_rng(0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_and_grad(w, b, X, y, lam=1e-3):
    """Mean binary cross-entropy with L2 penalty, and its gradient (derived by hand)."""
    p = sigmoid(X @ w + b)                                   # forward pass
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 * lam * w @ w
    r = (p - y) / len(y)                                     # dLoss/dz for every sample
    return loss, X.T @ r + lam * w, r.sum()                  # chain rule: dz/dw = x, dz/db = 1

def fit(X, y, lr=0.5, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        _, gw, gb = loss_and_grad(w, b, X, y)
        w, b = w - lr * gw, b - lr * gb                      # gradient descent
    return w, b

# Synthetic "pass / fail" data: e.g. whether a design meets a requirement, from 2 normalized features
X = rng.normal(size=(500, 2))
y = (X @ np.array([2.0, -1.0]) + 0.5 + 0.5 * rng.normal(size=500) > 0).astype(float)

w0, b0 = rng.normal(size=2), 0.1                             # gradient check at a random point
_, gw, gb = loss_and_grad(w0, b0, X, y)
h = 1e-6
num = [(loss_and_grad(w0 + h * e, b0, X, y)[0] - loss_and_grad(w0 - h * e, b0, X, y)[0]) / (2 * h) for e in np.eye(2)]
grad_check = np.max(np.abs(np.array(num) - gw))
print("gradient check, max difference:", grad_check)

w, b = fit(X, y)
acc = np.mean((sigmoid(X @ w + b) > 0.5) == y)
print(f"learned weights {np.round(w, 2)}, bias {b:.2f}, training accuracy {acc:.3f}")

from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(C=1 / (1e-3 * len(y))).fit(X, y)   # same penalty, scikit-learn's parametrization
print(f"scikit-learn weights {np.round(sk.coef_[0], 2)}, bias {sk.intercept_[0]:.2f}")
```

Output:

```
gradient check, max difference: 2.345276750581604e-11
learned weights [ 5.88 -2.87], bias 1.22, training accuracy 0.936
scikit-learn weights [ 5.89 -2.88], bias 1.22
```

Three habits shown here are what interviewers look for: derive the gradient, **check it numerically** with central differences before trusting it, and **compare with a reference implementation**. (scikit-learn's C is the inverse of the total penalty, so C = 1/(λN) matches λ in the mean loss.)

## Backpropagation in a neural network

A network is a chain of simple operations. For a two-layer network with input X:

```
forward:   z1 = X W1 + b1  →  a1 = tanh(z1)  →  out = a1 W2 + b2  →  L = mean((out − y)²)

backward:  dL/dout = 2(out − y)/N
           dL/dW2  = a1ᵀ · dL/dout          dL/db2 = sum of dL/dout
           dL/da1  = dL/dout · W2ᵀ
           dL/dz1  = dL/da1 ⊙ (1 − a1²)      (derivative of tanh, elementwise)
           dL/dW1  = Xᵀ · dL/dz1            dL/db1 = sum of dL/dz1
```

**The pattern for every layer:** receive the gradient with respect to the layer's output; compute the gradient with respect to its parameters (to update them) and with respect to its input (to pass back to the previous layer). A linear layer y = xW sends back (∂L/∂y)Wᵀ and computes ∂L/∂W = xᵀ(∂L/∂y); an elementwise activation multiplies by its derivative.

**Why it is efficient.** Each intermediate gradient is computed once and reused by everything upstream, so the backward pass costs a small constant times the forward pass, for millions of parameters at once. This is reverse-mode automatic differentiation; see [Automatic differentiation](automatic-differentiation.md#reverse-mode). The cost is memory: the forward intermediates (z1, a1) must be stored for the backward pass, which is what [gradient checkpointing](scaling-to-large-meshes.md#gradient-checkpointing) trades against recomputation.

## Code: a backward pass by hand, checked against autograd

```python
import torch

torch.manual_seed(0)
X = torch.randn(64, 3)
y = torch.randn(64, 1)
W1 = torch.randn(3, 16, requires_grad=True); b1 = torch.zeros(16, requires_grad=True)
W2 = torch.randn(16, 1, requires_grad=True); b2 = torch.zeros(1, requires_grad=True)

z1 = X @ W1 + b1                     # forward pass, keeping the intermediates
a1 = torch.tanh(z1)
out = a1 @ W2 + b2
loss = ((out - y) ** 2).mean()
loss.backward()                      # PyTorch's reverse-mode autodiff

with torch.no_grad():                # the same backward pass, by hand, layer by layer
    d_out = 2 * (out - y) / len(y)   # dL/d(out)
    dW2, db2 = a1.T @ d_out, d_out.sum(0)
    d_a1 = d_out @ W2.T              # push the gradient back through the linear layer
    d_z1 = d_a1 * (1 - a1 ** 2)      # through tanh: tanh'(z) = 1 - tanh(z)^2
    dW1, db1 = X.T @ d_z1, d_z1.sum(0)

match = all(torch.allclose(mine, p.grad, atol=1e-6) for mine, p in [(dW1, W1), (db1, b1), (dW2, W2), (db2, b2)])
print("hand-written backward pass matches autograd:", match)
```

Output:

```
hand-written backward pass matches autograd: True
```

## Practical details interviewers ask about

- **Numerical stability.** Compute the sigmoid and the log together in a stable way (libraries use log-sum-exp forms such as `BCEWithLogitsLoss`), instead of taking the log of a sigmoid that has underflowed to zero.
- **Vanishing and exploding gradients.** In deep networks, gradients are products of many local derivatives; saturating activations (sigmoid, tanh in their flat regions) shrink them, large weights grow them. Remedies: good initialization (Xavier, He), normalization layers, residual connections, gradient clipping.
- **Learning rate and feature scaling.** Unscaled features make the loss landscape elongated and gradient descent slow or unstable; standardize inputs first. See [Normalizing fields and coordinates](nondimensionalization-and-normalization.md#normalizing-fields-and-coordinates).
- **Mini-batches.** The gradient over a batch is an unbiased estimate of the full gradient; stochastic gradient descent and Adam use it. See [Optimizers for physics models](optimizers-for-physics-models.md).
- **Gradient checking.** Central differences on a few random coordinates, in double precision, comparing relative errors; essential when writing custom layers or physics losses.

## In an interview

**Thirty-second answer to "Explain backpropagation."** "Backpropagation is the chain rule applied backwards through the computation graph. The forward pass computes and stores intermediate values; the backward pass starts from the derivative of the loss with respect to the output and, layer by layer, multiplies by each operation's local derivative, giving the gradient for that layer's parameters and the gradient to pass to the layer before. Every intermediate gradient is reused, so the cost is about one extra forward pass regardless of the number of parameters, at the price of storing activations."

**Likely follow-ups**

- *What is the gradient of logistic regression?* Xᵀ(p − y)/N plus the regularization term; the sigmoid and log cancel to give p − y.
- *How do you test a gradient implementation?* Central finite differences on a few coordinates, compared with relative error, then compare with a library.
- *Why is logistic regression convex but a neural network not?* The loss is convex in the linear score and the score is linear in the parameters; hidden layers make the score nonlinear in the parameters.

## References

- [Neural Networks: Zero to Hero, lecture 1 on backpropagation (Andrej Karpathy, video)](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [micrograd: a tiny autograd engine (Andrej Karpathy)](https://github.com/karpathy/micrograd)
- [Automatic differentiation in machine learning: a survey (Baydin et al.)](https://arxiv.org/abs/1502.05767)
- [CS231n notes: backpropagation, intuitions (Stanford)](https://cs231n.github.io/optimization-2/)
