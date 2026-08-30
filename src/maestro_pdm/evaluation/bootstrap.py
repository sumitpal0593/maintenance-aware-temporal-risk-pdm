from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int = 5000
    confidence_level: float = 0.95
    seed: int = 42


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("Labels and scores must have the same length.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes are required.")
    if not np.isfinite(s).all():
        raise ValueError("Scores contain non-finite values.")

    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, s))
    return {
        "auroc": float(roc_auc_score(y, s)),
        "pr_auc": pr_auc,
        "enrichment": float(pr_auc / prevalence),
    }


def stratified_bootstrap_indices(
    y_true: np.ndarray,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Generate paired, class-stratified bootstrap index matrices.

    Each row contains one bootstrap sample. Positive and negative validation
    vehicles are sampled separately with replacement, preserving the original
    class counts in every resample. The same row of indices is then used for
    all models, which makes model comparisons paired.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes are required.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1.")

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    out = np.empty((n_resamples, len(y)), dtype=np.int32)
    for b in range(n_resamples):
        pos_b = rng.choice(pos, size=len(pos), replace=True)
        neg_b = rng.choice(neg, size=len(neg), replace=True)
        # Concatenation order is immaterial for ranking metrics, and preserving
        # a deterministic class block avoids an unnecessary shuffle.
        out[b] = np.concatenate([pos_b, neg_b]).astype(np.int32, copy=False)
    return out


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    model_scores: Mapping[str, np.ndarray],
    config: BootstrapConfig,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    scores = {
        name: np.asarray(values, dtype=float).reshape(-1)
        for name, values in model_scores.items()
    }
    for name, values in scores.items():
        if values.shape[0] != y.shape[0]:
            raise ValueError(f"Length mismatch for model {name!r}.")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite scores for model {name!r}.")

    point = {name: ranking_metrics(y, values) for name, values in scores.items()}
    indices = stratified_bootstrap_indices(y, config.n_resamples, config.seed)

    distributions = {
        name: {
            "auroc": np.empty(config.n_resamples, dtype=np.float64),
            "pr_auc": np.empty(config.n_resamples, dtype=np.float64),
            "enrichment": np.empty(config.n_resamples, dtype=np.float64),
        }
        for name in scores
    }

    for b, idx in enumerate(indices):
        yb = y[idx]
        for name, values in scores.items():
            mb = ranking_metrics(yb, values[idx])
            for metric in ("auroc", "pr_auc", "enrichment"):
                distributions[name][metric][b] = mb[metric]

    return point, distributions


def percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")
    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
