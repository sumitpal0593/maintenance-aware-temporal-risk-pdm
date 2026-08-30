from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DEFAULT_PROBABILITY_CLIP = 1e-6


@dataclass(frozen=True)
class PlattParameters:
    intercept: float
    slope: float
    C: float


def _as_binary_array(y) -> np.ndarray:
    arr = np.asarray(y, dtype=int).reshape(-1)
    unique = np.unique(arr)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(
            f"Expected both binary classes {{0,1}}, found {unique.tolist()}."
        )
    return arr


def clip_probabilities(
    probabilities,
    eps: float = DEFAULT_PROBABILITY_CLIP,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if not 0 < float(eps) < 0.5:
        raise ValueError("eps must lie in (0, 0.5).")
    return np.clip(p, float(eps), 1.0 - float(eps))


def logit_from_probability(
    probabilities,
    eps: float = DEFAULT_PROBABILITY_CLIP,
) -> np.ndarray:
    p = clip_probabilities(probabilities, eps=eps)
    return np.log(p / (1.0 - p))


def analytical_weight_correction(
    weighted_probability,
    omega_0: float,
    omega_1: float,
    eps: float = DEFAULT_PROBABILITY_CLIP,
) -> np.ndarray:
    """
    Invert the class-weighted BCE probability distortion:

        r = omega_0 q /
            [omega_1 (1-q) + omega_0 q]

    where q is the raw sigmoid output learned under weighted BCE.

    This is algebraically equivalent to shifting the raw logit by
    log(omega_0 / omega_1).
    """
    if omega_0 <= 0 or omega_1 <= 0:
        raise ValueError("Class weights must be strictly positive.")

    q = clip_probabilities(weighted_probability, eps=eps)
    denominator = omega_1 * (1.0 - q) + omega_0 * q
    corrected = (omega_0 * q) / denominator
    return clip_probabilities(corrected, eps=eps)


def analytical_weight_correction_from_logits(
    raw_logits,
    omega_0: float,
    omega_1: float,
    eps: float = DEFAULT_PROBABILITY_CLIP,
) -> np.ndarray:
    logits = np.asarray(raw_logits, dtype=float).reshape(-1)
    shifted = logits + np.log(float(omega_0) / float(omega_1))
    p = 1.0 / (1.0 + np.exp(-shifted))
    return clip_probabilities(p, eps=eps)


def fit_platt_scaler(
    raw_logits,
    y_true,
    C: float = 1e6,
    seed: int = 42,
) -> LogisticRegression:
    """
    Fit held-out Platt scaling on the frozen model's raw logits.

    No class weighting is used in the calibration regression.
    """
    logits = np.asarray(raw_logits, dtype=float).reshape(-1, 1)
    y = _as_binary_array(y_true)

    if len(logits) != len(y):
        raise ValueError("raw_logits and y_true must have equal length.")

    model = LogisticRegression(
        solver="lbfgs",
        C=float(C),
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(logits, y)
    return model


def apply_platt_scaler(
    model: LogisticRegression,
    raw_logits,
    eps: float = DEFAULT_PROBABILITY_CLIP,
) -> np.ndarray:
    logits = np.asarray(raw_logits, dtype=float).reshape(-1, 1)
    p = model.predict_proba(logits)[:, 1]
    return clip_probabilities(p, eps=eps)


def platt_parameters(
    model: LogisticRegression,
) -> PlattParameters:
    return PlattParameters(
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
        C=float(model.C),
    )


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> float:
    """
    Standard equal-width Expected Calibration Error (ECE).
    """
    y = _as_binary_array(y_true)
    p = clip_probabilities(probabilities)

    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have equal length.")

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # Include p=1 in the final bin.
    bin_idx = np.digitize(p, edges[1:-1], right=False)

    ece = 0.0
    n = len(y)

    for b in range(int(n_bins)):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue

        mean_p = float(p[mask].mean())
        event_rate = float(y[mask].mean())
        ece += (count / n) * abs(mean_p - event_rate)

    return float(ece)


def diagnostic_calibration_intercept_slope(
    y_true,
    probabilities,
    C: float = 1e6,
) -> tuple[float, float]:
    """
    Estimate calibration intercept and slope for diagnostic reporting only.

    A logistic regression of Y on logit(predicted probability) is fit using
    evaluation labels. The returned coefficients must NOT be used to modify
    or recalibrate those same evaluation predictions.
    """
    y = _as_binary_array(y_true)
    score = logit_from_probability(probabilities).reshape(-1, 1)

    model = LogisticRegression(
        solver="lbfgs",
        C=float(C),
        max_iter=2000,
    )
    model.fit(score, y)

    return (
        float(model.intercept_[0]),
        float(model.coef_[0, 0]),
    )


def probability_metrics(
    y_true,
    probabilities,
    ece_bins: int = 10,
) -> dict:
    """
    Ranking + probability-quality metrics for one prediction method.
    """
    y = _as_binary_array(y_true)
    p = clip_probabilities(probabilities)

    intercept, slope = diagnostic_calibration_intercept_slope(y, p)

    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, p))

    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "positive_prevalence": prevalence,
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": pr_auc,
        "enrichment": float(pr_auc / prevalence),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_equal_width": expected_calibration_error(
            y,
            p,
            n_bins=int(ece_bins),
        ),
        "diagnostic_calibration_intercept": intercept,
        "diagnostic_calibration_slope": slope,
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
    }


def probability_quantiles(
    probabilities,
    method: str,
) -> pd.DataFrame:
    p = clip_probabilities(probabilities)

    quantile_levels = [
        ("p01", 0.01),
        ("p05", 0.05),
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
    ]

    rows = [
        {"method": method, "statistic": "min", "value": float(np.min(p))},
        {"method": method, "statistic": "mean", "value": float(np.mean(p))},
        {"method": method, "statistic": "std", "value": float(np.std(p))},
    ]

    rows.extend(
        {
            "method": method,
            "statistic": name,
            "value": float(np.quantile(p, q)),
        }
        for name, q in quantile_levels
    )

    rows.append(
        {"method": method, "statistic": "max", "value": float(np.max(p))}
    )
    return pd.DataFrame(rows)


def reliability_table_quantile(
    y_true,
    probabilities,
    method: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Equal-frequency reliability table.

    Quantile bins are used for the plot so each bin remains informative under
    severe class imbalance and strongly skewed probability distributions.
    """
    y = _as_binary_array(y_true)
    p = clip_probabilities(probabilities)

    frame = pd.DataFrame({"y": y, "p": p})
    unique_count = int(frame["p"].nunique())
    q = min(int(n_bins), unique_count)

    if q < 2:
        return pd.DataFrame(
            [
                {
                    "method": method,
                    "bin": 1,
                    "count": len(frame),
                    "min_probability": float(frame["p"].min()),
                    "max_probability": float(frame["p"].max()),
                    "mean_probability": float(frame["p"].mean()),
                    "observed_event_rate": float(frame["y"].mean()),
                }
            ]
        )

    frame["bin"] = pd.qcut(
        frame["p"],
        q=q,
        labels=False,
        duplicates="drop",
    )

    rows = []
    for bin_id, group in frame.groupby("bin", sort=True, observed=True):
        rows.append(
            {
                "method": method,
                "bin": int(bin_id) + 1,
                "count": int(len(group)),
                "min_probability": float(group["p"].min()),
                "max_probability": float(group["p"].max()),
                "mean_probability": float(group["p"].mean()),
                "observed_event_rate": float(group["y"].mean()),
            }
        )

    return pd.DataFrame(rows)


def threshold_support_table(
    probabilities,
    method: str,
    target_levels: Iterable[float] = (0.10, 0.20, 0.30, 0.50, 0.70),
) -> pd.DataFrame:
    """
    Show whether illustrative risk levels are empirically represented.

    This is useful for replacing hypothetical 0.30/0.50/0.70 examples with
    values supported by the actual calibrated validation distribution.
    """
    p = clip_probabilities(probabilities)
    rows = []

    for target in target_levels:
        target = float(target)
        idx = int(np.argmin(np.abs(p - target)))
        nearest = float(p[idx])

        rows.append(
            {
                "method": method,
                "target_probability": target,
                "nearest_observed_probability": nearest,
                "absolute_distance": abs(nearest - target),
                "count_at_or_above_target": int((p >= target).sum()),
                "fraction_at_or_above_target": float((p >= target).mean()),
            }
        )

    return pd.DataFrame(rows)
