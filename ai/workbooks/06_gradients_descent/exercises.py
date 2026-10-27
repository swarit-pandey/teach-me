"""Workbook 06 — Partials, gradients & gradient descent.

Rules of the game
  * Points x are 1-D NumPy float arrays of shape (d,). A scalar function f takes
    such an x and returns a float; a gradient function returns an array of shape (d,).
  * Part A: gradients — numerical and analytic.
  * Part B: gradient descent, and the learning-rate bound 2 / lambda_max on quadratics.
  * Part C (stretch): the Hessian by finite differences, and what it says about
    critical points.
  * Never mutate an input array (copy first). Run   ./course check 06   to grade.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — gradients
# ----------------------------------------------------------------------------

def numerical_gradient(f, x, h=1e-5):
    """Central-difference gradient of scalar f at x (any array shape; returns
    the same shape). Component i is (f(x + h e_i) - f(x - h e_i)) / (2h).

    Must NOT modify x. Hint: loop over np.ndindex(x.shape) and perturb a copy.
    """
    raise NotImplementedError


def directional_derivative(f, x, u, h=1e-5):
    """Rate of change of f at x in the direction of u (u need not be unit length:
    normalize it first). Equals  grad f(x) . u_hat  (lesson §3).
    Use numerical_gradient. Raise ValueError if u is the zero vector.
    """
    raise NotImplementedError


def quad_form_grad(A, x):
    """Gradient of the quadratic form q(x) = x^T A x, for a square (NOT
    necessarily symmetric) matrix A.

    Hint: derive it component-wise: q = sum_ij A_ij x_i x_j. The answer is not
    2Ax unless A is symmetric.
    """
    raise NotImplementedError


def rosenbrock(x):
    """f(x, y) = (1 - x)^2 + 100 (y - x^2)^2  for x = np.array([x, y])."""
    raise NotImplementedError


def rosenbrock_grad(x):
    """Analytic gradient of rosenbrock, shape (2,). Minimum is at (1, 1)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — gradient descent
# ----------------------------------------------------------------------------

def gradient_descent(grad, x0, lr, steps):
    """Run x <- x - lr * grad(x) for `steps` steps starting at x0.

    Returns the history as an array of shape (k + 1, d): row 0 is x0, row k the
    last iterate. Normally k == steps. Stop EARLY (and return what you have) as
    soon as an iterate is non-finite or has max |x_i| > 1e12 — that's divergence;
    include that last iterate in the history.
    """
    raise NotImplementedError


def max_stable_lr(A):
    """For f(x) = 1/2 x^T A x - b^T x with A symmetric positive definite, gradient
    descent converges from every start iff 0 < lr < 2 / lambda_max(A). Return
    that bound. Hint: np.linalg.eigvalsh returns eigenvalues in ascending order.
    """
    raise NotImplementedError


def optimal_lr_and_rate(A):
    """Return (lr*, rate*) for gradient descent on the quadratic with SPD matrix A.

    Each eigen-direction's error is multiplied by (1 - lr * lambda_i) per step, so
    the worst-case rate is max_i |1 - lr * lambda_i|. It is minimized when the
    extreme eigenvalues balance:  lr* = 2 / (lambda_min + lambda_max), giving
    rate* = (kappa - 1) / (kappa + 1) with kappa = lambda_max / lambda_min.
    """
    raise NotImplementedError


def gd_converges(A, b, lr, steps=2000, tol=1e-6):
    """Empirically test whether gradient descent on f(x) = 1/2 x^T A x - b^T x,
    started at x0 = 0, ends within `tol` (Euclidean distance) of the minimizer
    x* = A^{-1} b after `steps` steps. Return a bool.
    Use your gradient_descent; the gradient is A x - b.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: second order
# ----------------------------------------------------------------------------

def hessian_fd(f, x, h=1e-4):
    """Hessian of scalar f at x (shape (d, d)) by finite differences.

    Entry (i, j) ~ [f(x+h e_i+h e_j) - f(x+h e_i-h e_j) - f(x-h e_i+h e_j) + f(x-h e_i-h e_j)] / (4h^2).
    Return a symmetric matrix (symmetrize with (H + H.T) / 2 to remove noise).
    """
    raise NotImplementedError


def classify_critical_point(H, tol=1e-8):
    """Given the Hessian at a point where the gradient is zero, return one of
    "minimum" (all eigenvalues > tol), "maximum" (all < -tol), "saddle" (some
    > tol and some < -tol), or "degenerate" (otherwise: some eigenvalue ~ 0, the
    second-order test can't decide).
    """
    raise NotImplementedError
