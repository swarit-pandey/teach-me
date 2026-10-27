"""Workbook 16 — Neural networks & MLPs.

Rules of the game
  * NumPy only. Batches are ROWS: an input batch X has shape (N, d_in), a weight
    matrix W has shape (d_in, d_out), so a layer is  Z = X @ W + b  with b of shape
    (d_out,) broadcast over the batch (lesson 12).
  * Part A: activations, the "linear layers collapse" fact, init, batched forward
    pass, parameter counting.
  * Part B: a 2-layer network with a HAND-DERIVED backward pass (lesson 07's
    Jacobians, lesson 14's softmax gradient), trained on the spirals dataset.
  * Part C (stretch): build a ReLU network by hand that exactly reproduces any
    piecewise-linear curve — the constructive heart of universal approximation.
  * Run   ./course check 16   to grade. helpers.py has make_spirals() and
    numerical_grad() — use them freely.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import numpy as np

from helpers import make_spirals  # noqa: F401  (handy in a REPL)


# ----------------------------------------------------------------------------
# Part A — building blocks
# ----------------------------------------------------------------------------

def relu(z):
    """Element-wise max(0, z). Works on any array shape."""
    raise NotImplementedError


def relu_grad(z):
    """Derivative of relu at z, element-wise: 1.0 where z > 0, else 0.0.
    (At exactly 0 the derivative is undefined; use 0 like every framework does.)"""
    raise NotImplementedError


def sigmoid(z):
    """Numerically stable logistic function 1 / (1 + exp(-z)).

    Must not overflow or return nan for z = ±1000.
    Hint: for z < 0 use the equivalent form exp(z) / (1 + exp(z)); np.where
    evaluates both branches, so compute exp(-|z|) once and build both from it.
    """
    raise NotImplementedError


def tanh_grad(z):
    """Derivative of tanh at z: 1 - tanh(z)^2."""
    raise NotImplementedError


def collapse_linear(layers):
    """Stacked linear layers with NO nonlinearity are one linear layer.

    layers: list of (W, b) with shapes (d0, d1), (d1,), then (d1, d2), (d2,), ...
    Return (W_eff, b_eff) such that for every X of shape (N, d0)

        ((X @ W1 + b1) @ W2 + b2) ... == X @ W_eff + b_eff.

    Hint: push one layer through at a time.  (X W1 + b1) W2 + b2 = X (W1 W2) + (b1 W2 + b2).
    """
    raise NotImplementedError


def count_params(sizes):
    """Number of trainable scalars in an MLP with layer widths `sizes`.

    sizes = [d0, d1, ..., dL]; layer l has a (d_{l-1}, d_l) weight matrix and a
    d_l bias. count_params([784, 128, 10]) == 101770.
    """
    raise NotImplementedError


def init_mlp(sizes, seed=0, scheme="he"):
    """Initialize an MLP. Returns a list of (W, b) tuples, one per layer.

    W has shape (d_in, d_out); b is zeros of shape (d_out,).
      scheme="he":    W ~ Normal(0, std = sqrt(2 / d_in))   (lesson 18 derives why)
      scheme="small": W ~ Normal(0, std = 0.01)
    Use np.random.default_rng(seed) and draw the layers in order.
    Raise ValueError for an unknown scheme.
    """
    raise NotImplementedError


def mlp_forward(params, X):
    """Forward pass of an MLP: ReLU after every layer EXCEPT the last.

    params: list of (W, b) as from init_mlp.  X: (N, d0).
    Returns (logits, hiddens): logits of shape (N, dL), and the list of
    post-ReLU hidden activations (one (N, d_l) array per hidden layer, in order).
    Hint: one loop over layers; no loop over examples — the batch rides along in X.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — a 2-layer net with a hand-written backward pass
# ----------------------------------------------------------------------------
# params is a dict {"W1": (D,H), "b1": (H,), "W2": (H,C), "b2": (C,)}.
#   forward:  z1 = X W1 + b1;  a1 = relu(z1);  logits = a1 W2 + b2

def softmax_cross_entropy(logits, y):
    """Mean cross-entropy loss and its gradient w.r.t. the logits.

    logits: (N, C); y: (N,) integer labels.
    Returns (loss, dlogits) with dlogits of shape (N, C).
    Hint: subtract the row max before exp (lesson 14). The gradient of the MEAN
    loss is (softmax - onehot(y)) / N.
    """
    raise NotImplementedError


def init_2layer(D, H, C, seed=0):
    """He-initialized params dict for a D -> H -> C network (zeros for biases).
    Draw W1 then W2 from np.random.default_rng(seed)."""
    raise NotImplementedError


def forward_2layer(params, X):
    """Return (logits, cache). Put whatever backward needs in cache (a dict)."""
    raise NotImplementedError


def backward_2layer(params, cache, dlogits):
    """Backpropagate dlogits = dL/dlogits to every parameter.

    Return a dict of gradients with the same keys and shapes as params.
    Hint (derive each line yourself first, lesson 07):
        dW2 = a1^T dlogits          db2 = sum over the batch of dlogits
        da1 = dlogits W2^T          dz1 = da1 * relu'(z1)
        dW1 = X^T dz1               db1 = sum over the batch of dz1
    """
    raise NotImplementedError


def train_2layer(X, y, H=100, lr=1.0, steps=3000, reg=1e-3, seed=0):
    """Full-batch gradient descent on loss = CE + (reg/2) * (||W1||^2 + ||W2||^2).

    Returns (params, losses) where losses[i] is the total loss at step i
    (before that step's update). Biases are not regularized.
    """
    raise NotImplementedError


def accuracy(params, X, y):
    """Fraction of rows whose argmax logit equals y."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: universal approximation, constructively
# ----------------------------------------------------------------------------

def relu_interpolant(xs, ys):
    """Build a 1-hidden-layer ReLU net f(x) = sum_k a_k relu(w_k x + c_k) + d that
    passes through every point (xs[i], ys[i]) and is linear in between.

    xs: strictly increasing array of n+1 knots; ys: values at the knots.
    Return (w, c, a, d) with w, c, a arrays of length n and d a float.
    On [xs[0], xs[-1]], f must equal np.interp(x, xs, ys).

    Hint: f(x) = y0 + s0 relu(x - x0) + sum_{k>=1} (s_k - s_{k-1}) relu(x - x_k),
    where s_k is the slope of segment k. Each unit is a "hinge" that bends the line.
    """
    raise NotImplementedError


def relu_net_eval(w, c, a, d, x):
    """Evaluate the net from relu_interpolant on an array x (any shape)."""
    raise NotImplementedError
