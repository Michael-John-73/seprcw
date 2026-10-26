"""Training on development data and image-level statistical summaries.

These utilities never load data or choose an experimental split. The orchestrator
must supply development-fit observations to fit_logistic, and freeze the model
before reading calibration or test outcomes.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import beta

FEATURE_NAMES = ["pilot_correlation", "reference_soft_match", "phase_margin", "coverage"]


def feature_vector(result: dict, reference) -> np.ndarray:
    """Reference is the public, precommitted payload, never an oracle crop label."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    soft = np.asarray(result["soft"], dtype=np.float64).reshape(-1)
    if not len(ref) or soft.shape != ref.shape or not np.isin(ref, [0, 1]).all():
        raise ValueError("Soft bits and binary reference must have matching nonzero lengths")
    values = np.array([float(result.get("pilot_score", 0.0)),
                       float(np.mean(soft * (2.0 * ref - 1.0))),
                       float(result.get("phase_margin", 0.0)),
                       float(result.get("coverage", 0.0))], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite decoder features must be recorded as failures, not silently fitted")
    return values


def _matrix(X) -> np.ndarray:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Expected finite N x 4 feature matrix")
    return x


def fit_logistic(X, y, lr: float = .1, steps: int = 300, l2: float = .01) -> dict:
    """Deterministic, balanced, standardized L2 logistic regression (four weights).

    This is detector-score training, not watermark encoder or generator training.
    Both marked and unmarked examples must come only from development-fit images.
    """
    x = _matrix(X)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(target) != len(x) or not np.isin(target, [0, 1]).all():
        raise ValueError("Labels must be aligned binary targets")
    if len(np.unique(target)) != 2:
        raise ValueError("Both positive and negative development examples are required")
    if not 0 < lr <= 10 or steps < 1 or l2 < 0:
        raise ValueError("Invalid optimization parameters")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = (x - mean) / scale
    w, bias = np.zeros(x.shape[1]), 0.0
    counts = np.bincount(target.astype(int), minlength=2)
    sample_weight = len(target) / (2.0 * counts[target.astype(int)])
    history = []
    for step in range(int(steps)):
        logit = z @ w + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        error = (probability - target) * sample_weight
        loss = float(np.mean(sample_weight * (np.logaddexp(0, logit) - target * logit))
                     + .5 * l2 * np.dot(w, w))
        if step == 0 or step % 25 == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss})
        w -= lr * (z.T @ error / len(target) + l2 * w)
        bias -= lr * float(error.mean())
    if not np.isfinite(w).all() or not math.isfinite(bias):
        raise ValueError("Training diverged; retain failure and change only development settings")
    return {"feature_names": FEATURE_NAMES, "weights": w.tolist(), "bias": bias,
            "mean": mean.tolist(), "scale": scale.tolist(), "lr": lr, "steps": int(steps),
            "l2": l2, "n_training_rows": len(target), "loss_history": history}


def predict_scores(X, model: dict) -> np.ndarray:
    x = _matrix(X)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if mean.shape != (4,) or scale.shape != (4,) or weights.shape != (4,) or (scale <= 0).any():
        raise ValueError("Invalid frozen score model")
    return ((x - mean) / scale) @ weights + float(model["bias"])


def conformal_threshold(scores, alpha: float) -> dict:
    """Strict score > threshold; None means +infinity and no acceptance.

    One score per independent source image, using the maximum over specified crop
    conditions AFTER the complete pilot-based phase search and score procedure.
    The exchangeability guarantee is marginal,
    not a guarantee for every fixed calibration set or every adversarial input.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < alpha < 1 or not len(s) or not np.isfinite(s).all():
        raise ValueError("Need finite nonempty image-level null scores and 0 < alpha < 1")
    rank = int(math.ceil((len(s) + 1) * (1 - alpha)))
    feasible = rank <= len(s)
    threshold = float(np.sort(s)[rank - 1]) if feasible else None
    return {"threshold": threshold, "feasible": feasible, "rank": rank,
            "n": len(s), "alpha": alpha, "rule": "strict_greater_than"}


def paired_bootstrap(a, b, seed: int = 123, reps: int = 2000,
                     confidence: float = .9833333333333333) -> dict:
    """Resample paired source-image means; crop variants are not independent N."""
    x, y = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Paired finite nonempty source-image vectors required")
    if not 0 < confidence < 1 or reps < 100:
        raise ValueError("Invalid interval configuration")
    diff = x - y
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    # Bound memory even when a large confirmatory cohort is used.
    for start in range(0, reps, 200):
        count = min(200, reps - start)
        indices = rng.integers(0, len(diff), size=(count, len(diff)))
        means[start:start + count] = diff[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return {"estimate": float(diff.mean()), "low": float(lo), "high": float(hi),
            "n": len(diff), "confidence": confidence, "resamples": reps,
            "unit": "source_image", "interval": "paired_percentile_bootstrap"}


def binomial_ci(k: int, n: int, confidence: float = .9833333333333333,
                one_sided: bool = True) -> dict:
    """Exact Clopper-Pearson interval for image-level any-false-accept events."""
    if n <= 0 or k < 0 or k > n or int(k) != k or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts/confidence")
    if one_sided:
        lo = 0.0
        hi = 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))
    else:
        tail = (1 - confidence) / 2
        lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return {"rate": k / n, "low": lo, "high": hi, "n": n, "k": int(k),
            "confidence": confidence, "one_sided": one_sided,
            "unit": "source_image", "interval": "Clopper-Pearson"}
