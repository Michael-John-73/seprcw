"""Source-cluster uncertainty; never treat crop/bit/seed rows as independent n."""

from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta, binom, norm


def cp_upper(x, n, gamma):
    if not (n > 0 and 0 <= x <= n and 0 < gamma < 1):
        raise ValueError("invalid exact binomial inputs")
    return 1.0 if x == n else float(beta.ppf(1 - gamma, x + 1, n - x))


def cp_lower(x, n, gamma):
    if not (n > 0 and 0 <= x <= n and 0 < gamma < 1):
        raise ValueError("invalid exact binomial inputs")
    return 0.0 if x == 0 else float(beta.ppf(gamma, x, n - x + 1))


def order_threshold(values, alpha):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or len(a) == 0 or np.isnan(a).any() or np.isposinf(a).any():
        raise ValueError("null scores must be nonempty, finite or rejection=-inf")
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    k = math.ceil((len(a) + 1) * (1 - alpha))
    return (float("inf") if k > len(a) else float(np.partition(a, k - 1)[k - 1])), k


def bootstrap_difference(d, reps, family_alpha, comparisons=6, seed=0):
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("need >=2 finite independent source differences")
    from .compute import cuda_enabled
    if cuda_enabled():
        from .cuda_statistics import bootstrap_difference as cuda_bootstrap
        return cuda_bootstrap(d, reps, family_alpha, comparisons, seed)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    # Bounded temporary memory rather than one reps*n allocation.
    chunk = max(1, min(256, 1_000_000 // len(d)))
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        draws[start : start + count] = d[
            rng.integers(len(d), size=(count, len(d)))
        ].mean(1)
    tail = family_alpha / (2 * comparisons)
    if np.ptp(d) == 0:
        # A point-mass empirical bootstrap cannot certify zero population uncertainty.
        radius = math.sqrt(2 * math.log(1 / tail) / len(d))
        return (
            float(d.mean()),
            max(-1.0, float(d.mean() - radius)),
            min(1.0, float(d.mean() + radius)),
        )
    return float(d.mean()), *map(float, np.quantile(draws, [tail, 1 - tail]))


def verdict(low, high, margin):
    if margin is None or margin <= 0:
        return "WITHHELD"
    if low > margin:
        return "SUPPORTED"
    if high <= margin:
        return "MARGIN_NOT_MET"
    return "INCONCLUSIVE"


def exact_fpr_power(n, assumed_p, alpha_test, gamma):
    if not 0 <= assumed_p <= 1:
        raise ValueError("assumed probability")
    # CP upper is monotone in x; binary-search largest admissible count.
    lo, hi = -1, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cp_upper(mid, n, gamma) <= alpha_test:
            lo = mid
        else:
            hi = mid
    return {
        "n": n,
        "max_false_accepts": lo,
        "assumed_p": assumed_p,
        "power": float(binom.cdf(lo, n, assumed_p)) if lo >= 0 else 0.0,
    }


def planning_effect(margin):
    if not 0 < margin < 1:
        raise ValueError("margin must be strictly between 0 and 1")
    return margin + min(margin, (1 - margin) / 2)


def planning_n(sd, margin, power, family_alpha, comparisons=6):
    """Normal planning scenario with bounded assumed effect; no performance guarantee."""
    if not (sd > 0 and margin > 0 and 0 < power < 1):
        raise ValueError("invalid planning inputs")
    z = norm.ppf(1 - family_alpha / (2 * comparisons)) + norm.ppf(power)
    gap = planning_effect(margin) - margin
    return max(2, math.ceil((z * sd / gap) ** 2))
