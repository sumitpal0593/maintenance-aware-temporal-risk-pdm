from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd

from maestro_pdm.config import get_output_dir, load_config
from maestro_pdm.evaluation.reference_baselines import (
    binary_ranking_metrics,
    load_reference_baseline_data,
    save_single_model_result,
    select_best_by_tuning,
    step_survival_array_risk_at_tau,
)


RSF_GRID = [
    {
        "n_estimators": 300,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "min_samples_split": 20,
        "min_samples_leaf": 30,
        "max_features": "sqrt",
    },
]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_experiment.yaml")
    return p.parse_args()


def make_y(Surv, durations, events):
    return Surv.from_arrays(
        event=np.asarray(events, dtype=bool),
        time=np.asarray(durations, dtype=float),
    )


def risk_at_tau(model, X, tau):
    surv = model.predict_survival_function(
        X,
        return_array=True,
    )
    return step_survival_array_risk_at_tau(
        surv,
        model.unique_times_,
        tau,
    )


def main():
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError(
            "scikit-survival is required. Install with: "
            "pip install scikit-survival"
        ) from exc

    a = args()
    cfg = load_config(a.config)
    output_root = get_output_dir(cfg)
    tau = float(cfg["experiment"]["tau"])
    seed = int(cfg["experiment"]["seed"])

    data = load_reference_baseline_data(output_root)
    y_surv = make_y(
        Surv,
        data.duration_train,
        data.event_train,
    )

    candidates = {}
    rows = []

    for hp in RSF_GRID:
        config = "|".join(
            f"{k}={hp[k]}"
            for k in sorted(hp)
        )
        print(f"Fitting RSF: {config}")

        model = RandomSurvivalForest(
            n_estimators=hp["n_estimators"],
            min_samples_split=hp["min_samples_split"],
            min_samples_leaf=hp["min_samples_leaf"],
            max_features=hp["max_features"],
            n_jobs=-1,
            random_state=seed,
            low_memory=False,
        )
        model.fit(data.X_train, y_surv)

        tune_risk = risk_at_tau(
            model,
            data.X_tuning,
            tau,
        )
        m = binary_ranking_metrics(
            data.y_tuning_horizon,
            tune_risk,
        )
        rows.append(
            {
                "model": "RSF",
                "implementation": "sksurv.ensemble.RandomSurvivalForest",
                "configuration": config,
                "status": "ok",
                "tuning_auroc": m["auroc"],
                "tuning_pr_auc": m["pr_auc"],
                "tuning_enrichment": m["enrichment"],
            }
        )
        candidates[config] = model

    tuning = pd.DataFrame(rows)
    best = select_best_by_tuning(tuning)
    config = str(best["configuration"])
    model = candidates[config]

    val_risk = risk_at_tau(
        model,
        data.X_validation,
        tau,
    )
    val_metrics = binary_ranking_metrics(
        data.y_validation_horizon,
        val_risk,
    )

    ckpt = output_root / "checkpoints" / "survival_baselines_reference"
    ckpt.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ckpt / "rsf_sksurv.joblib")

    selected = {
        "model": "RSF",
        "implementation": "sksurv.ensemble.RandomSurvivalForest",
        "configuration": config,
        "selection_metric": "vehicle-level tuning PR-AUC",
        "tau": tau,
        "feature_dimension": data.X_train.shape[1],
        "training_samples": len(data.X_train),
    }

    save_single_model_result(
        output_root,
        "rsf",
        tuning,
        selected,
        val_metrics,
        data.validation_vehicle_id,
        data.y_validation_horizon,
        val_risk,
    )

    print("\nRSF REFERENCE-IMPLEMENTATION RESULTS")
    print("------------------------------------")
    print(f"Selected configuration : {config}")
    print(f"Validation AUROC       : {val_metrics['auroc']:.6f}")
    print(f"Validation PR-AUC      : {val_metrics['pr_auc']:.6f}")
    print(f"Validation enrichment  : {val_metrics['enrichment']:.6f}")


if __name__ == "__main__":
    main()
