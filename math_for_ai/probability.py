"""Bernoulli experiments and Monte Carlo estimation."""
import numpy as np


def bernoulli_trial_mean(p, n, seed=0):
    """Return sample mean and estimated standard error from n Bernoulli draws."""
    p = float(p)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1')
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('n must be an integer >= 2')
    draws = np.random.default_rng(seed).binomial(1, p, size=n)
    return float(draws.mean()), float(draws.std(ddof=1) / np.sqrt(n))


def theoretical_standard_error(p, n):
    """Exact standard deviation of the Bernoulli sample mean."""
    p = float(p)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1')
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    return float(np.sqrt(p * (1 - p) / n))
