from __future__ import annotations

import argparse
from copy import deepcopy

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
    survival_df_risk_at_tau,
)


LEARNING_RATE_GRID = [1e-3, 3e-4]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_experiment.yaml")
    return p.parse_args()


def main():
    try:
        import torchtuples as tt
        from pycox.models import CoxPH
    except ImportError as exc:
        raise ImportError(
            "pycox and torchtuples are required. Install with:\n"
            "pip install pycox torchtuples"
        ) from exc

    a = args()
    cfg = load_config(a.config)
    output_root = get_output_dir(cfg)
    tau = float(cfg["experiment"]["tau"])
    seed = int(cfg["experiment"]["seed"])

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_reference_baseline_data(output_root)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(data.X_train).astype(np.float32)
    Xtu = scaler.transform(data.X_tuning).astype(np.float32)
    Xva = scaler.transform(data.X_validation).astype(np.float32)

    candidates = {}
    rows = []

    for lr in LEARNING_RATE_GRID:
        torch.manual_seed(seed)
        config = (
            f"hidden=128,128|dropout=0.1|lr={lr:g}|"
            "batch_size=1024|max_epochs=30|patience=5"
        )
        print(f"Fitting Deep CoxPH (pycox): {config}")

        net = tt.practical.MLPVanilla(
            Xtr.shape[1],
            [128, 128],
            1,
            batch_norm=True,
            dropout=0.10,
        )
        model = CoxPH(
            net,
            tt.optim.AdamW(lr=float(lr)),
        )

        callbacks = [
            tt.callbacks.EarlyStopping(patience=5)
        ]

        log = model.fit(
            Xtr,
            (
                data.duration_train.astype(np.float32),
                data.event_train.astype(np.int64),
            ),
            batch_size=1024,
            epochs=30,
            callbacks=callbacks,
            val_data=(
                Xtu,
                (
                    data.duration_tuning.astype(np.float32),
                    data.event_tuning.astype(np.int64),
                ),
            ),
            verbose=False,
        )

        model.compute_baseline_hazards(
            Xtr,
            (
                data.duration_train.astype(np.float32),
                data.event_train.astype(np.int64),
            ),
        )

        tune_surv = model.predict_surv_df(Xtu)
        tune_risk = survival_df_risk_at_tau(
            tune_surv,
            tau,
        )
        m = binary_ranking_metrics(
            data.y_tuning_horizon,
            tune_risk,
        )

        hist_path = (
            output_root / "tables"
            / (
                "deep_coxph_pycox_history_lr_"
                + str(lr).replace(".", "p")
                + ".csv"
            )
        )
        try:
            log.to_pandas().to_csv(
                hist_path,
                index=False,
            )
        except Exception:
            pass

        rows.append(
            {
                "model": "Deep CoxPH",
                "implementation": "pycox.models.CoxPH",
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

    val_surv = model.predict_surv_df(Xva)
    val_risk = survival_df_risk_at_tau(
        val_surv,
        tau,
    )
    val_metrics = binary_ranking_metrics(
        data.y_validation_horizon,
        val_risk,
    )

    ckpt = output_root / "checkpoints" / "survival_baselines_reference"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.net.state_dict(),
            "configuration": config,
            "input_dim": int(Xtr.shape[1]),
            "hidden": [128, 128],
            "dropout": 0.10,
            "tau": tau,
        },
        ckpt / "deep_coxph_pycox.pt",
    )
    joblib.dump(
        scaler,
        ckpt / "deep_coxph_scaler.joblib",
    )

    selected = {
        "model": "Deep CoxPH",
        "implementation": "pycox.models.CoxPH",
        "configuration": config,
        "selection_metric": "vehicle-level tuning PR-AUC",
        "tau": tau,
        "feature_dimension": data.X_train.shape[1],
        "training_samples": len(data.X_train),
    }

    save_single_model_result(
        output_root,
        "deep_coxph",
        tuning,
        selected,
        val_metrics,
        data.validation_vehicle_id,
        data.y_validation_horizon,
        val_risk,
    )

    print("\nDEEP COXPH REFERENCE-IMPLEMENTATION RESULTS")
    print("-------------------------------------------")
    print(f"Selected configuration : {config}")
    print(f"Validation AUROC       : {val_metrics['auroc']:.6f}")
    print(f"Validation PR-AUC      : {val_metrics['pr_auc']:.6f}")
    print(f"Validation enrichment  : {val_metrics['enrichment']:.6f}")


if __name__ == "__main__":
    main()
