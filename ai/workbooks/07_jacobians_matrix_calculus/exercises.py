"""Workbook 07 — Jacobians & matrix calculus.

Rules of the game
  * NumPy throughout. Vectors are 1-D arrays; batches are 2-D arrays with one
    example per ROW (shape (N, d)), as in every deep-learning library.
  * The shape rule: the gradient of a scalar loss w.r.t. a parameter has exactly
    the parameter's shape. Check shapes first whenever you're stuck.
  * Part A: Jacobians — numerical and analytic.
  * Part B: gradients of scalar losses w.r.t. vectors and matrices, and a full
    manual backward pass through a 2-layer network, verified by gradient checking.
  * Part C (stretch): forward vs reverse mode — same product, different cost.
  * Run   ./course check 07   to grade.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — Jacobians
# ----------------------------------------------------------------------------

def numerical_jacobian(f, x, h=1e-5):
    """Central-difference Jacobian of f: R^n -> R^m at x (1-D array, shape (n,)).

    f returns a 1-D array of shape (m,). Return J of shape (m, n) with
    J[i, j] = d f_i / d x_j. Column j is (f(x + h e_j) - f(x - h e_j)) / (2h).
    Must not modify x.
    """
    raise NotImplementedError


def jac_affine(A, b, x):
    """Jacobian of x -> A x + b at x. (Yes, it's short. Know why.)"""
    raise NotImplementedError


def jac_elementwise(fprime, x):
    """Jacobian of x -> (f(x_1), ..., f(x_n)) for a scalar function f applied
    elementwise, given its derivative fprime (vectorized). Shape (n, n).
    Hint: output i depends only on input i.
    """
    raise NotImplementedError


def softmax(z):
    """Numerically stable softmax of a 1-D array: subtract max(z) before exp."""
    raise NotImplementedError


def jac_softmax(z):
    """Jacobian of softmax at z:  J = diag(p) - p p^T  where p = softmax(z).

    Derive it (lesson §5): d p_i / d z_j = p_i (delta_ij - p_j).
    """
    raise NotImplementedError


def chain_jacobian(jac_f, jac_g, f, x):
    """Jacobian of g(f(x)) at x, given callables jac_f(x) -> (m, n) and
    jac_g(u) -> (k, m). Chain rule: J_g(f(x)) @ J_f(x). Shape (k, n).
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — gradients of scalar losses; backprop by hand
# ----------------------------------------------------------------------------

def grad_least_squares(A, x, b):
    """Gradient w.r.t. x of  L(x) = ||A x - b||^2  (no 1/2!). Shape (n,).
    Answer: 2 A^T (A x - b). Derive it via the chain rule with r = Ax - b.
    """
    raise NotImplementedError


def grad_linear_layer(W, x, y):
    """For L(W) = 1/2 ||W x - y||^2 with W of shape (m, n), x (n,), y (m,),
    return dL/dW, shape (m, n).

    Hint: with delta = W x - y (shape (m,)), dL/dW_ij = delta_i x_j: an outer product.
    """
    raise NotImplementedError


def init_params(d_in, d_hidden, d_out, seed=0):
    """Return dict with W1 (d_in, d_hidden), b1 (d_hidden,), W2 (d_hidden, d_out),
    b2 (d_out,). Weights ~ N(0, 1/fan_in) i.e. scale 1/sqrt(fan_in); biases 0.
    Use np.random.default_rng(seed).
    """
    raise NotImplementedError


def forward(params, X):
    """Two-layer network on a batch X of shape (N, d_in):

        Z1 = X @ W1 + b1        (N, d_hidden)
        H  = tanh(Z1)           (N, d_hidden)
        Yh = H @ W2 + b2        (N, d_out)

    Return (Yh, cache) where cache holds whatever backward() needs
    (at least X and H).
    """
    raise NotImplementedError


def mse_loss(Yh, Y):
    """L = 1/(2N) * sum of squared errors over the whole (N, d_out) batch."""
    raise NotImplementedError


def backward(params, cache, Yh, Y):
    """Gradients of mse_loss(forward(params, X)[0], Y) w.r.t. every parameter.

    Return a dict with keys W1, b1, W2, b2, each the same shape as the parameter.
    Walk the graph backwards, one local Jacobian at a time:
        dYh = (Yh - Y) / N
        dW2 = H^T dYh,   db2 = sum over the batch of dYh,   dH = dYh W2^T
        dZ1 = dH * (1 - H^2)          (tanh' = 1 - tanh^2, elementwise)
        dW1 = X^T dZ1,   db1 = sum over the batch of dZ1
    """
    raise NotImplementedError


def grad_check(loss_fn, params, grads, h=1e-6):
    """Compare analytic `grads` with central differences of loss_fn(params)
    (a function of the whole params dict) for EVERY entry of every parameter.

    Return the maximum relative error over all entries, where
        rel = |a - n| / max(1e-12, |a| + |n|).
    Restore every entry you perturb. (A correct backward pass gives ~1e-8 or less.)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: forward vs reverse mode
# ----------------------------------------------------------------------------

def jacobian_product(Js, mode):
    """Multiply a chain of Jacobians  J_L @ ... @ J_2 @ J_1  (Js = [J_1, ..., J_L],
    J_k of shape (n_k, n_{k-1})) and count the multiply-adds.

    mode="forward": start from J_1 and multiply on the LEFT by J_2, J_3, ...
                    (what you'd get pushing n_0 tangent vectors forward).
    mode="reverse": start from J_L and multiply on the RIGHT by J_{L-1}, ...
                    (what you'd get pulling n_L cotangent vectors backward).
    Multiplying (a, b) @ (b, c) costs a*b*c multiply-adds.
    Return (product, cost). Raise ValueError for an unknown mode.
    """
    raise NotImplementedError
