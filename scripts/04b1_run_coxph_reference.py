from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from maestro_pdm.config import get_output_dir, load_config
from maestro_pdm.evaluation.reference_baselines import (
    binary_ranking_metrics,
    load_reference_baseline_data,
    save_single_model_result,
    select_best_by_tuning,
)


PENALIZER_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_experiment.yaml")
    return p.parse_args()


def feature_frame(X, duration=None, event=None):
    columns = [f"x_{j:04d}" for j in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    if duration is not None:
        df["duration"] = np.asarray(duration, dtype=float)
    if event is not None:
        df["event"] = np.asarray(event, dtype=int)
    return df


def risk_at_tau(cph, X_df, tau):
    sf = cph.predict_survival_function(
        X_df,
        times=[float(tau)],
    )
    return np.clip(
        1.0 - sf.iloc[0].to_numpy(dtype=float),
        0.0,
        1.0,
    )


def main():
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError(
            "lifelines is required. Install with: pip install lifelines"
        ) from exc

    a = args()
    cfg = load_config(a.config)
    output_root = get_output_dir(cfg)
    tau = float(cfg["experiment"]["tau"])

    data = load_reference_baseline_data(output_root)

    # ------------------------------------------------------------------
    # CoxPH-specific stability filter.
    #
    # lifelines internally standardizes covariates. A feature that is
    # constant (or numerically almost constant) in model_train therefore
    # produces a zero/near-zero standard deviation and can make the Newton
    # system singular. The filter is fitted on model_train only and then
    # applied unchanged to tuning and public validation.
    # ------------------------------------------------------------------
    raw_std = np.nanstd(data.X_train.astype(np.float64), axis=0)
    finite_mask = np.all(np.isfinite(data.X_train), axis=0)
    variance_mask = raw_std > 1e-10
    keep_mask = finite_mask & variance_mask

    if not np.any(keep_mask):
        raise RuntimeError("No CoxPH covariates remain after variance filtering.")

    dropped_idx = np.flatnonzero(~keep_mask)
    kept_idx = np.flatnonzero(keep_mask)

    print(
        f"CoxPH variance filter      : kept {len(kept_idx)} / "
        f"{data.X_train.shape[1]} features; dropped {len(dropped_idx)}"
    )
    if len(dropped_idx):
        print(
            "Dropped zero/near-zero variance feature indices: "
            + ", ".join(map(str, dropped_idx[:30]))
            + (" ..." if len(dropped_idx) > 30 else "")
        )

    X_train_filtered = data.X_train[:, keep_mask]
    X_tuning_filtered = data.X_tuning[:, keep_mask]
    X_validation_filtered = data.X_validation[:, keep_mask]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_filtered)
    Xtu = scaler.transform(X_tuning_filtered)
    Xva = scaler.transform(X_validation_filtered)

    # Defensive finite-value check after transformation.
    if not (
        np.all(np.isfinite(Xtr))
        and np.all(np.isfinite(Xtu))
        and np.all(np.isfinite(Xva))
    ):
        raise RuntimeError(
            "Non-finite values remain after CoxPH variance filtering/scaling."
        )

    train_df = feature_frame(
        Xtr,
        data.duration_train,
        data.event_train,
    )
    tune_X = feature_frame(Xtu)
    val_X = feature_frame(Xva)

    candidates = {}
    rows = []

    for penalizer in PENALIZER_GRID:
        config = f"penalizer={penalizer:g}|l1_ratio=0"
        print(f"Fitting CoxPH: {config}")

        try:
            cph = CoxPHFitter(
                penalizer=float(penalizer),
                l1_ratio=0.0,
            )
            cph.fit(
                train_df,
                duration_col="duration",
                event_col="event",
                show_progress=False,
                fit_options={
                    "step_size": 0.5,
                    "max_steps": 500,
                    "precision": 1e-8,
                },
            )

            tune_risk = risk_at_tau(cph, tune_X, tau)
            m = binary_ranking_metrics(
                data.y_tuning_horizon,
                tune_risk,
            )
            rows.append(
                {
                    "model": "CoxPH",
                    "implementation": "lifelines.CoxPHFitter",
                    "configuration": config,
                    "status": "ok",
                    "tuning_auroc": m["auroc"],
                    "tuning_pr_auc": m["pr_auc"],
                    "tuning_enrichment": m["enrichment"],
                }
            )
            candidates[config] = cph

        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {ex}")
            rows.append(
                {
                    "model": "CoxPH",
                    "implementation": "lifelines.CoxPHFitter",
                    "configuration": config,
                    "status": "failed",
                    "error": repr(ex),
                    "tuning_auroc": np.nan,
                    "tuning_pr_auc": np.nan,
                    "tuning_enrichment": np.nan,
                }
            )

    tuning = pd.DataFrame(rows)
    best = select_best_by_tuning(tuning)
    config = str(best["configuration"])
    model = candidates[config]

    val_risk = risk_at_tau(model, val_X, tau)
    val_metrics = binary_ranking_metrics(
        data.y_validation_horizon,
        val_risk,
    )

    ckpt = output_root / "checkpoints" / "survival_baselines_reference"
    ckpt.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "keep_mask": keep_mask,
            "kept_feature_indices": kept_idx,
            "dropped_feature_indices": dropped_idx,
            "variance_threshold": 1e-10,
            "scaler": scaler,
            "model": model,
        },
        ckpt / "coxph_lifelines.joblib",
    )

    pd.DataFrame(
        {
            "original_feature_index": np.arange(data.X_train.shape[1]),
            "training_std": raw_std,
            "finite_in_model_train": finite_mask,
            "kept_for_coxph": keep_mask,
        }
    ).to_csv(
        output_root / "tables" / "coxph_variance_filter_audit.csv",
        index=False,
    )

    selected = {
        "model": "CoxPH",
        "implementation": "lifelines.CoxPHFitter",
        "configuration": config,
        "selection_metric": "vehicle-level tuning PR-AUC",
        "tau": tau,
        "feature_dimension_before_filter": data.X_train.shape[1],
        "feature_dimension_after_filter": int(np.sum(keep_mask)),
        "variance_filter_threshold": 1e-10,
        "training_samples": len(data.X_train),
    }

    save_single_model_result(
        output_root,
        "coxph",
        tuning,
        selected,
        val_metrics,
        data.validation_vehicle_id,
        data.y_validation_horizon,
        val_risk,
    )

    print("\nCOXPH REFERENCE-IMPLEMENTATION RESULTS")
    print("--------------------------------------")
    print(f"Selected configuration : {config}")
    print(f"Validation AUROC       : {val_metrics['auroc']:.6f}")
    print(f"Validation PR-AUC      : {val_metrics['pr_auc']:.6f}")
    print(f"Validation enrichment  : {val_metrics['enrichment']:.6f}")


if __name__ == "__main__":
    main()
