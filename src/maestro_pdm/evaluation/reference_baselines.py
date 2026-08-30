from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from maestro_pdm.evaluation.survival_utils import (
    build_survival_target_frame,
)


@dataclass
class ReferenceBaselineData:
    X_train: np.ndarray
    X_tuning: np.ndarray
    X_validation: np.ndarray
    train_vehicle_id: np.ndarray
    tune_vehicle_id: np.ndarray
    validation_vehicle_id: np.ndarray
    train_prefix_time_step: np.ndarray
    tune_prefix_time_step: np.ndarray
    duration_train: np.ndarray
    event_train: np.ndarray
    duration_tuning: np.ndarray
    event_tuning: np.ndarray
    y_train_horizon: np.ndarray
    y_tuning_horizon: np.ndarray
    y_validation_horizon: np.ndarray


def binary_ranking_metrics(y_true, scores) -> dict:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)

    if len(y) != len(s):
        raise ValueError("Labels and scores must have the same length.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes are required.")

    prevalence = float(np.mean(y))
    pr_auc = float(average_precision_score(y, s))
    return {
        "n": int(len(y)),
        "n_positive": int(np.sum(y)),
        "positive_prevalence": prevalence,
        "auroc": float(roc_auc_score(y, s)),
        "pr_auc": pr_auc,
        "enrichment": float(pr_auc / prevalence),
    }


def load_npz(path: Path) -> dict:
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def load_reference_baseline_data(output_root: Path) -> ReferenceBaselineData:
    features_dir = output_root / "features"
    tables_dir = output_root / "tables"

    required = [
        features_dir / "temporal_control_train_features.npz",
        features_dir / "temporal_control_tuning_features.npz",
        features_dir / "temporal_control_validation_features.npz",
        tables_dir / "horizon_prefix_samples.csv",
        tables_dir / "tuning_vehicle_landmarks.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 4A outputs are required before Stage 4B:\n  - "
            + "\n  - ".join(missing)
        )

    train = load_npz(required[0])
    tune = load_npz(required[1])
    val = load_npz(required[2])

    X_train = np.concatenate(
        [train["aggregate_dynamic"], train["static"]],
        axis=1,
    ).astype(np.float32)
    X_tune = np.concatenate(
        [tune["aggregate_dynamic"], tune["static"]],
        axis=1,
    ).astype(np.float32)
    X_val = np.concatenate(
        [val["aggregate_dynamic"], val["static"]],
        axis=1,
    ).astype(np.float32)

    horizon_samples = pd.read_csv(required[3])
    tuning_landmarks = pd.read_csv(required[4])

    train_surv = build_survival_target_frame(
        train["vehicle_id"],
        train["prefix_time_step"],
        horizon_samples,
        source_split="model_train",
    )
    tune_surv = build_survival_target_frame(
        tune["vehicle_id"],
        tune["prefix_time_step"],
        tuning_landmarks,
        source_split="tuning",
    )

    y_train = train["label"].astype(int)
    y_tune = tune["label"].astype(int)
    y_val = val["binary_label"].astype(int)

    if not np.array_equal(
        train_surv["label"].to_numpy(dtype=int),
        y_train,
    ):
        raise RuntimeError("Training horizon labels are misaligned.")
    if not np.array_equal(
        tune_surv["label"].to_numpy(dtype=int),
        y_tune,
    ):
        raise RuntimeError("Tuning horizon labels are misaligned.")

    if X_train.shape[1] != X_tune.shape[1] or X_train.shape[1] != X_val.shape[1]:
        raise RuntimeError("Feature dimensions differ across splits.")

    return ReferenceBaselineData(
        X_train=X_train,
        X_tuning=X_tune,
        X_validation=X_val,
        train_vehicle_id=train["vehicle_id"].astype(np.int64),
        tune_vehicle_id=tune["vehicle_id"].astype(np.int64),
        validation_vehicle_id=val["vehicle_id"].astype(np.int64),
        train_prefix_time_step=train["prefix_time_step"].astype(np.int64),
        tune_prefix_time_step=tune["prefix_time_step"].astype(np.int64),
        duration_train=train_surv["remaining_time_steps"].to_numpy(float),
        event_train=train_surv["event_observed"].to_numpy(int),
        duration_tuning=tune_surv["remaining_time_steps"].to_numpy(float),
        event_tuning=tune_surv["event_observed"].to_numpy(int),
        y_train_horizon=y_train,
        y_tuning_horizon=y_tune,
        y_validation_horizon=y_val,
    )


def select_best_by_tuning(tuning_df: pd.DataFrame) -> pd.Series:
    valid = tuning_df[
        tuning_df["status"].eq("ok")
        & tuning_df["tuning_pr_auc"].notna()
    ].copy()

    if valid.empty:
        raise RuntimeError("No valid candidate model completed successfully.")

    valid = valid.sort_values(
        ["tuning_pr_auc", "tuning_auroc", "configuration"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return valid.iloc[0]


def survival_df_risk_at_tau(surv_df: pd.DataFrame, tau: float) -> np.ndarray:
    """
    Convert a survival-function dataframe (time index x samples) to
    horizon risk 1-S(tau). Uses the last survival value at time <= tau.
    """
    times = np.asarray(surv_df.index, dtype=float)
    idx = int(np.searchsorted(times, float(tau), side="right") - 1)

    if idx < 0:
        return np.zeros(surv_df.shape[1], dtype=float)

    idx = min(idx, len(times) - 1)
    survival = surv_df.iloc[idx].to_numpy(dtype=float)
    return np.clip(1.0 - survival, 0.0, 1.0)


def step_survival_array_risk_at_tau(
    survival_array: np.ndarray,
    unique_times,
    tau: float,
) -> np.ndarray:
    times = np.asarray(unique_times, dtype=float)
    surv = np.asarray(survival_array, dtype=float)

    idx = int(np.searchsorted(times, float(tau), side="right") - 1)
    if idx < 0:
        return np.zeros(surv.shape[0], dtype=float)

    idx = min(idx, surv.shape[1] - 1)
    return np.clip(1.0 - surv[:, idx], 0.0, 1.0)


def save_single_model_result(
    output_root: Path,
    model_slug: str,
    tuning_df: pd.DataFrame,
    selected_row: dict,
    validation_metrics: dict,
    validation_vehicle_id,
    y_validation,
    validation_risk,
) -> None:
    tables = output_root / "tables"
    predictions = output_root / "predictions"
    logs = output_root / "logs"

    for p in (tables, predictions, logs):
        p.mkdir(parents=True, exist_ok=True)

    tuning_df.to_csv(
        tables / f"{model_slug}_reference_tuning.csv",
        index=False,
    )
    pd.DataFrame([selected_row]).to_csv(
        tables / f"{model_slug}_reference_selected.csv",
        index=False,
    )

    metrics_row = {
        "model": selected_row["model"],
        "implementation": selected_row["implementation"],
        "configuration": selected_row["configuration"],
        **validation_metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(
        tables / f"{model_slug}_reference_validation_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "vehicle_id": np.asarray(validation_vehicle_id, dtype=np.int64),
            "binary_label": np.asarray(y_validation, dtype=int),
            "risk_tau": np.asarray(validation_risk, dtype=float),
        }
    ).to_csv(
        predictions / f"{model_slug}_reference_validation_predictions.csv",
        index=False,
    )
