"""Explicit gradient descent updates and numerical traces."""
import numpy as np
from .linear_algebra import _array


def _steps(steps):
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")


def _positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def descent_regime(eta, kappa):
    """Classify the exact-real-arithmetic update factor of a positive quadratic.

    The labels describe nonzero starting points; x0=0 is always a fixed point.
    Floating-point inputs are classified by their computed product.
    """
    p = _positive(eta, 'eta') * _positive(kappa, 'kappa')
    if p < 1:
        return 'direct convergence'
    if p == 1:
        return 'one-step convergence'
    if p < 2:
        return 'oscillatory convergence'
    if p == 2:
        return 'constant-amplitude oscillation'
    return 'divergence'


def quadratic_descent(x0, eta, kappa, steps):
    """Return arrays of iterates and costs, including the initial state."""
    _steps(steps)
    eta, kappa = _positive(eta, 'eta'), _positive(kappa, 'kappa')
    x0 = float(x0)
    if not np.isfinite(x0):
        raise ValueError('x0 must be finite')
    x = np.empty(steps + 1)
    x[0] = x0
    with np.errstate(over='raise', invalid='raise'):
        factor = np.float64(1) - np.float64(eta) * kappa
        for i in range(steps):
            x[i+1] = factor * x[i]
        costs = (kappa / 2) * x**2
    return x, costs


def linear_regression_descent(X, y, eta, steps, initial=None):
    """Minimize ||Xw-y||²/(2n) using full-batch gradient descent.

    Include a column of ones in X if an intercept is desired. Return the
    weight trajectory and objective values. No feature scaling is performed.
    """
    X, y = _array(X, 2), _array(y, 1)
    _steps(steps)
    eta = _positive(eta, 'eta')
    if X.shape[0] != y.size:
        raise ValueError('Row count must match target length')
    w = np.zeros(X.shape[1]) if initial is None else _array(initial, 1).copy()
    if w.size != X.shape[1]:
        raise ValueError('Initial weight dimension must match column count')
    weights, losses = [], []
    with np.errstate(over='raise', invalid='raise'):
        for i in range(steps + 1):
            residual = X @ w - y
            weights.append(w.copy())
            losses.append(float(residual @ residual / (2 * y.size)))
            if i < steps:
                w -= eta * (X.T @ residual / y.size)
    return np.asarray(weights), np.asarray(losses)
