from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from maestro_pdm.config import get_output_dir, load_config
from maestro_pdm.evaluation.reference_baselines import (
    binary_ranking_metrics,
    load_reference_baseline_data,
    save_single_model_result,
    select_best_by_tuning,
)


MIXTURE_GRID = [2, 3]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_experiment.yaml")
    return p.parse_args()


def main():
    try:
        from auton_survival.models.dsm import DeepSurvivalMachines
    except ImportError as exc:
        raise ImportError(
            "auton-survival is required for the reference DSM. "
            "The original notebook used a Python 3.10 conda environment. "
            "Install with: pip install auton-survival"
        ) from exc

    a = args()
    cfg = load_config(a.config)
    output_root = get_output_dir(cfg)
    tau = float(cfg["experiment"]["tau"])
    seed = int(cfg["experiment"]["seed"])

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_reference_baseline_data(output_root)

    # Training-only standardization is applied consistently in the revised
    # protocol. This avoids validation-derived preprocessing while preserving
    # the original DSM implementation and hidden-layer family.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(data.X_train).astype(np.float64)
    Xtu = scaler.transform(data.X_tuning).astype(np.float64)
    Xva = scaler.transform(data.X_validation).astype(np.float64)

    candidates = {}
    rows = []

    for k in MIXTURE_GRID:
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = (
            f"k={k}|layers=64,32|iters=100|lr=0.001|batch_size=128"
        )
        print(f"Fitting DSM (auton-survival): {config}")

        dsm = DeepSurvivalMachines(
            k=int(k),
            layers=[64, 32],
        )
        dsm.fit(
            Xtr,
            data.duration_train.astype(np.float64),
            data.event_train.astype(np.int64),
            iters=100,
            learning_rate=1e-3,
            batch_size=128,
        )

        # Preserve the dtype fix that was required by the original notebook.
        if hasattr(dsm, "torch_model"):
            dsm.torch_model = dsm.torch_model.double()

        tune_survival = dsm.predict_survival(
            Xtu,
            t=float(tau),
        )
        tune_risk = 1.0 - np.asarray(
            tune_survival,
            dtype=np.float64,
        ).reshape(-1)

        m = binary_ranking_metrics(
            data.y_tuning_horizon,
            tune_risk,
        )
        rows.append(
            {
                "model": "DSM",
                "implementation": (
                    "auton_survival.models.dsm.DeepSurvivalMachines"
                ),
                "configuration": config,
                "status": "ok",
                "tuning_auroc": m["auroc"],
                "tuning_pr_auc": m["pr_auc"],
                "tuning_enrichment": m["enrichment"],
            }
        )
        candidates[config] = dsm

    tuning = pd.DataFrame(rows)
    best = select_best_by_tuning(tuning)
    config = str(best["configuration"])
    model = candidates[config]

    val_survival = model.predict_survival(
        Xva,
        t=float(tau),
    )
    val_risk = 1.0 - np.asarray(
        val_survival,
        dtype=np.float64,
    ).reshape(-1)

    val_metrics = binary_ranking_metrics(
        data.y_validation_horizon,
        val_risk,
    )

    ckpt = output_root / "checkpoints" / "survival_baselines_reference"
    ckpt.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        ckpt / "dsm_scaler.joblib",
    )

    if hasattr(model, "torch_model"):
        torch.save(
            {
                "state_dict": model.torch_model.state_dict(),
                "configuration": config,
                "tau": tau,
            },
            ckpt / "dsm_auton_survival.pt",
        )

    selected = {
        "model": "DSM",
        "implementation": (
            "auton_survival.models.dsm.DeepSurvivalMachines"
        ),
        "configuration": config,
        "selection_metric": "vehicle-level tuning PR-AUC",
        "tau": tau,
        "feature_dimension": data.X_train.shape[1],
        "training_samples": len(data.X_train),
    }

    save_single_model_result(
        output_root,
        "dsm",
        tuning,
        selected,
        val_metrics,
        data.validation_vehicle_id,
        data.y_validation_horizon,
        val_risk,
    )

    print("\nDSM REFERENCE-IMPLEMENTATION RESULTS")
    print("------------------------------------")
    print(f"Selected configuration : {config}")
    print(f"Validation AUROC       : {val_metrics['auroc']:.6f}")
    print(f"Validation PR-AUC      : {val_metrics['pr_auc']:.6f}")
    print(f"Validation enrichment  : {val_metrics['enrichment']:.6f}")


if __name__ == "__main__":
    main()
