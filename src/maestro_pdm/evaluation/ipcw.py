from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class IPCWDiagnostics:
    n_landmarks_for_censoring_model: int
    n_censoring_events: int
    n_observed_events: int
    censor_survival_before_tau: float
    raw_weight_min: float
    raw_weight_median: float
    raw_weight_mean: float
    raw_weight_max: float
    normalized_weight_min: float
    normalized_weight_max: float
    effective_sample_size: float
    unweighted_positive_prevalence: float
    ipcw_weighted_positive_prevalence: float


def build_censoring_km(
    landmark_audit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate the censoring survival function G(u)=P(C >= u) from the complete
    set of selected calibration landmarks.

    For the censoring model:
      - an original right-censoring endpoint (event_observed=0) is treated as
        the event of interest;
      - an observed repair/failure endpoint (event_observed=1) is treated as
        censored for the censoring-time model.

    Only landmarks with a selected prefix and positive residual follow-up are
    used. Importantly, this includes landmarks whose Y(tau) is unknown because
    censoring occurred before tau; those observations are essential for
    estimating the censoring mechanism.
    """
    required = {
        "vehicle_id",
        "prefix_index",
        "remaining_time_steps",
        "event_observed",
    }
    missing = required.difference(landmark_audit.columns)
    if missing:
        raise ValueError(
            f"Landmark audit is missing columns: {sorted(missing)}"
        )

    frame = landmark_audit.copy()
    frame = frame[
        frame["prefix_index"].notna()
        & frame["remaining_time_steps"].notna()
        & frame["event_observed"].notna()
    ].copy()

    frame["remaining_time_steps"] = frame[
        "remaining_time_steps"
    ].astype(int)
    frame["event_observed"] = frame["event_observed"].astype(int)

    frame = frame[frame["remaining_time_steps"] > 0].copy()

    if frame.empty:
        raise ValueError(
            "No selected landmarks with positive residual follow-up are "
            "available for the censoring model."
        )
    if not set(frame["event_observed"].unique()).issubset({0, 1}):
        raise ValueError("event_observed must be binary.")

    times = frame["remaining_time_steps"].to_numpy(dtype=int)
    censor_events = (frame["event_observed"].to_numpy(dtype=int) == 0)

    survival = 1.0
    rows = []

    for t in np.sort(np.unique(times)):
        at_risk = int(np.sum(times >= t))
        d_censor = int(np.sum((times == t) & censor_events))
        d_failure = int(np.sum((times == t) & (~censor_events)))

        survival_before = survival
        if at_risk <= 0:
            raise RuntimeError("Invalid Kaplan-Meier risk set.")

        if d_censor > 0:
            survival = survival * (1.0 - d_censor / at_risk)

        rows.append(
            {
                "time": int(t),
                "n_at_risk": at_risk,
                "censoring_events": d_censor,
                "observed_failures": d_failure,
                "G_before": float(survival_before),
                "G_after": float(survival),
            }
        )

    return pd.DataFrame(rows)


def censor_survival_before(
    km_table: pd.DataFrame,
    query_times,
) -> np.ndarray:
    """
    Return G(u-) = P(C >= u), i.e. the Kaplan-Meier censoring survival just
    before each query time.

    Using G(u-) is appropriate in discrete time here because a subject censored
    exactly at the horizon boundary has still been observed event-free through
    that boundary.
    """
    required = {"time", "G_after"}
    missing = required.difference(km_table.columns)
    if missing:
        raise ValueError(f"KM table is missing columns: {sorted(missing)}")

    km_times = km_table["time"].to_numpy(dtype=float)
    g_after = km_table["G_after"].to_numpy(dtype=float)
    q = np.asarray(query_times, dtype=float).reshape(-1)

    out = np.ones_like(q, dtype=float)

    # Largest KM time strictly less than the query time.
    indices = np.searchsorted(km_times, q, side="left") - 1
    mask = indices >= 0
    out[mask] = g_after[indices[mask]]

    return out


def compute_horizon_ipcw_weights(
    labelled_landmarks: pd.DataFrame,
    km_table: pd.DataFrame,
    tau: int,
    numerical_floor: float = 1e-8,
) -> pd.DataFrame:
    """
    Compute fixed-horizon IPCW weights for known Y(tau) outcomes.

    Positive event within tau:
        w_i = 1 / G(R_i-)
        where R_i is the residual time from the landmark to the observed event.

    Known negative through tau:
        w_i = 1 / G(tau-)

    Raw weights are also normalized to mean one before fitting Platt scaling.
    This preserves relative IPCW contributions while avoiding an arbitrary
    change in the effective regularization strength of the logistic model.
    """
    tau = int(tau)
    if tau <= 0:
        raise ValueError("tau must be positive.")

    required = {
        "vehicle_id",
        "remaining_time_steps",
        "event_observed",
        "label",
        "status",
    }
    missing = required.difference(labelled_landmarks.columns)
    if missing:
        raise ValueError(
            f"Labelled landmark table is missing columns: {sorted(missing)}"
        )

    frame = labelled_landmarks.copy()
    frame = frame[frame["label"].notna()].copy()

    if frame.empty:
        raise ValueError("No labelled landmarks are available for IPCW.")

    frame["label"] = frame["label"].astype(int)
    frame["remaining_time_steps"] = frame[
        "remaining_time_steps"
    ].astype(int)

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be binary.")

    g_tau = float(
        censor_survival_before(
            km_table,
            np.asarray([tau], dtype=float),
        )[0]
    )
    g_tau = max(g_tau, float(numerical_floor))

    positive_mask = frame["label"].to_numpy(dtype=int) == 1
    g_used = np.full(len(frame), g_tau, dtype=float)

    if positive_mask.any():
        positive_times = frame.loc[
            positive_mask,
            "remaining_time_steps",
        ].to_numpy(dtype=float)

        g_positive = censor_survival_before(
            km_table,
            positive_times,
        )
        g_positive = np.maximum(
            g_positive,
            float(numerical_floor),
        )
        g_used[positive_mask] = g_positive

    raw_weights = 1.0 / g_used
    normalized_weights = raw_weights / np.mean(raw_weights)

    frame["censor_survival_G"] = g_used
    frame["ipcw_raw"] = raw_weights
    frame["ipcw_normalized"] = normalized_weights

    return frame


def effective_sample_size(weights) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    denominator = float(np.sum(w ** 2))
    if denominator <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / denominator)


def summarize_ipcw(
    full_landmark_audit: pd.DataFrame,
    weighted_labelled_landmarks: pd.DataFrame,
    km_table: pd.DataFrame,
    tau: int,
) -> IPCWDiagnostics:
    selected = full_landmark_audit[
        full_landmark_audit["prefix_index"].notna()
        & full_landmark_audit["remaining_time_steps"].notna()
        & full_landmark_audit["event_observed"].notna()
    ].copy()
    selected = selected[
        selected["remaining_time_steps"].astype(float) > 0
    ]

    w = weighted_labelled_landmarks["ipcw_normalized"].to_numpy(dtype=float)
    raw_w = weighted_labelled_landmarks["ipcw_raw"].to_numpy(dtype=float)
    y = weighted_labelled_landmarks["label"].to_numpy(dtype=float)

    g_tau = float(
        censor_survival_before(
            km_table,
            np.asarray([tau], dtype=float),
        )[0]
    )

    return IPCWDiagnostics(
        n_landmarks_for_censoring_model=int(len(selected)),
        n_censoring_events=int(
            (selected["event_observed"].astype(int) == 0).sum()
        ),
        n_observed_events=int(
            (selected["event_observed"].astype(int) == 1).sum()
        ),
        censor_survival_before_tau=g_tau,
        raw_weight_min=float(np.min(raw_w)),
        raw_weight_median=float(np.median(raw_w)),
        raw_weight_mean=float(np.mean(raw_w)),
        raw_weight_max=float(np.max(raw_w)),
        normalized_weight_min=float(np.min(w)),
        normalized_weight_max=float(np.max(w)),
        effective_sample_size=effective_sample_size(w),
        unweighted_positive_prevalence=float(np.mean(y)),
        ipcw_weighted_positive_prevalence=float(
            np.sum(w * y) / np.sum(w)
        ),
    )


def fit_ipcw_platt_scaler(
    raw_logits,
    y_true,
    sample_weight,
    C: float = 1e6,
    seed: int = 42,
) -> LogisticRegression:
    """
    Fit Platt scaling to frozen raw logits using IPCW sample weights.
    """
    logits = np.asarray(raw_logits, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)

    if not (len(logits) == len(y) == len(weights)):
        raise ValueError(
            "raw_logits, y_true and sample_weight must have equal length."
        )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes are required.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("IPCW sample weights must be finite and positive.")

    model = LogisticRegression(
        solver="lbfgs",
        C=float(C),
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(
        logits,
        y,
        sample_weight=weights,
    )
    return model
