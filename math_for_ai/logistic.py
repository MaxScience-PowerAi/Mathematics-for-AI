"""Binary logistic regression with NumPy and explicit gradients."""
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def _validate(X, y, w):
    X, y, w = (np.asarray(a, dtype=float) for a in (X, y, w))
    if X.ndim != 2 or not X.size or y.shape != (X.shape[0],) or w.shape != (X.shape[1],):
        raise ValueError('Incompatible dimensions or empty design matrix')
    if not all(np.isfinite(a).all() for a in (X, y, w)) or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite inputs and binary labels are required')
    return X, y, w


def loss_gradient(X, y, w, l2=0.0):
    """Mean binary cross-entropy plus l2*||w||²/2; regularizes all weights."""
    X, y, w = _validate(X, y, w)
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError('l2 must be finite and nonnegative')
    with np.errstate(over='raise', invalid='raise'):
        signed = (2*y - 1) * (X @ w)
        loss = np.logaddexp(0.0, -signed).mean() + l2/2 * (w @ w)
        grad = X.T @ (-(2*y - 1) * sigmoid(-signed)) / len(y) + l2*w
    return float(loss), grad


def fit(X, y, steps=200, l2=0.01):
    """Full-batch descent with a Lipschitz step; add a ones column for a bias."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be a matrix')
    X, y, w = _validate(X, y, np.zeros(X.shape[1]))
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    loss, grad = loss_gradient(X, y, w, l2)
    L = np.linalg.norm(X, ord=2)**2 / (4*len(y)) + l2
    if not np.isfinite(L):
        raise ValueError('Rescale X before fitting')
    history = [loss]
    if L == 0:
        return w, np.repeat(loss, steps+1)
    for _ in range(steps):
        w -= grad / L
        loss, grad = loss_gradient(X, y, w, l2)
        history.append(loss)
    return w, np.asarray(history)
