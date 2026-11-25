"""Numerically motivated linear algebra examples using NumPy."""
import numpy as np


def _array(value, ndim):
    a = np.asarray(value, dtype=float)
    if a.ndim != ndim or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected a nonempty finite {ndim}-dimensional array")
    return a


def cosine_similarity(x, y):
    """Cosine of the angle between two nonzero vectors of equal length."""
    x, y = _array(x, 1), _array(y, 1)
    if x.shape != y.shape:
        raise ValueError("Vector dimensions must agree")
    sx, sy = np.max(np.abs(x)), np.max(np.abs(y))
    if sx == 0 or sy == 0:
        raise ValueError("Cosine is undefined for a zero vector")
    x, y = x / sx, y / sy
    return float(np.clip((x / np.linalg.norm(x)) @ (y / np.linalg.norm(y)), -1, 1))


def project(x, onto):
    """Orthogonal projection of x onto the line spanned by onto."""
    x, u = _array(x, 1), _array(onto, 1)
    if x.shape != u.shape:
        raise ValueError("Vector dimensions must agree")
    s = np.max(np.abs(u))
    if s == 0:
        raise ValueError("Projection direction must be nonzero")
    u = u / s
    u = u / np.linalg.norm(u)
    with np.errstate(over='raise', invalid='raise'):
        return (x @ u) * u


def least_squares(A, b):
    """Return a minimum-norm minimizer of ||Ax-b||_2; supports rank deficiency."""
    A, b = _array(A, 2), _array(b, 1)
    if A.shape[0] != b.size:
        raise ValueError("Row count must match target length")
    return np.linalg.lstsq(A, b, rcond=None)[0]


def svd_approximation(A, rank):
    """Best rank-at-most-r approximation in Frobenius norm (truncated SVD)."""
    A = _array(A, 2)
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 0 <= rank <= min(A.shape):
        raise ValueError("rank must be an integer between 0 and min(A.shape)")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
