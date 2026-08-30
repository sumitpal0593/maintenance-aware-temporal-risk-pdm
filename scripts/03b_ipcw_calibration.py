from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from maestro_pdm.config import get_output_dir, load_config
from maestro_pdm.evaluation.calibration import (
    apply_platt_scaler,
    probability_metrics,
    probability_quantiles,
    reliability_table_quantile,
    threshold_support_table,
)
from maestro_pdm.evaluation.ipcw import (
    build_censoring_km,
    compute_horizon_ipcw_weights,
    fit_ipcw_platt_scaler,
    summarize_ipcw,
)
from maestro_pdm.reproducibility import set_global_seed
from maestro_pdm.visualization.calibration import (
    save_probability_histogram,
    save_reliability_diagram,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3.1: fit a censoring-aware IPCW Platt calibration map "
            "using only the held-out calibration landmark data produced in "
            "Stage 3. The frozen GRU is not retrained."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/paper_experiment.yaml",
    )
    return parser.parse_args()


def require_files(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Stage 3 files are missing. Run Stage 3 first:\n"
            "python scripts/03_calibrate_predictions.py "
            "--config configs/paper_experiment.yaml\n\nMissing:\n  - "
            + "\n  - ".join(missing)
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["experiment"]["seed"])
    tau = int(cfg["experiment"]["tau"])
    set_global_seed(seed)

    output_root = get_output_dir(cfg)
    tables_dir = output_root / "tables"
    predictions_dir = output_root / "predictions"
    checkpoints_dir = output_root / "checkpoints"
    figures_dir = output_root / "figures"
    logs_dir = output_root / "logs"

    for p in (
        tables_dir,
        predictions_dir,
        checkpoints_dir,
        figures_dir,
        logs_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    calibration_audit_path = (
        tables_dir / "calibration_vehicle_landmarks.csv"
    )
    calibration_predictions_path = (
        predictions_dir / "calibration_raw_predictions.csv"
    )
    validation_calibrated_path = (
        predictions_dir / "validation_calibrated_predictions.csv"
    )
    stage3_metrics_path = (
        tables_dir / "calibration_metrics_validation.csv"
    )

    require_files(
        [
            calibration_audit_path,
            calibration_predictions_path,
            validation_calibrated_path,
            stage3_metrics_path,
        ]
    )

    print("=" * 78)
    print("STAGE 3.1")
    print("Censoring-aware IPCW Platt calibration")
    print("=" * 78)
    print(f"Prediction horizon   : tau = {tau}")
    print("GRU retraining       : NO")
    print("Validation used to fit calibrator: NO")
    print()

    landmark_audit = pd.read_csv(calibration_audit_path)
    calibration_pred = pd.read_csv(calibration_predictions_path)
    validation = pd.read_csv(validation_calibrated_path)

    required_calibration_prediction_cols = {
        "vehicle_id",
        "prefix_time_step",
        "label",
        "raw_logit",
        "raw_probability",
    }
    missing = required_calibration_prediction_cols.difference(
        calibration_pred.columns
    )
    if missing:
        raise ValueError(
            "Calibration prediction table is missing columns: "
            f"{sorted(missing)}"
        )

    required_validation_cols = {
        "vehicle_id",
        "binary_label",
        "raw_probability",
        "analytical_probability",
        "platt_probability",
        "raw_logit",
    }
    missing = required_validation_cols.difference(validation.columns)
    if missing:
        raise ValueError(
            "Validation calibrated prediction table is missing columns: "
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------------------
    # 1. Estimate G(u) from ALL selected calibration landmarks, including
    #    censored-before-tau landmarks whose binary Y(tau) is unknown.
    # ------------------------------------------------------------------
    km_table = build_censoring_km(landmark_audit)
    km_table.to_csv(
        tables_dir / "ipcw_censoring_km.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Attach IPCW weights to the labelled calibration landmarks.
    # ------------------------------------------------------------------
    labelled_audit = landmark_audit[
        landmark_audit["label"].notna()
    ].copy()

    if labelled_audit["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Calibration landmark audit contains duplicate labelled vehicles."
        )
    if calibration_pred["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Calibration predictions contain duplicate vehicles."
        )

    weighted_landmarks = compute_horizon_ipcw_weights(
        labelled_landmarks=labelled_audit,
        km_table=km_table,
        tau=tau,
    )

    calibration_fit = calibration_pred.merge(
        weighted_landmarks[
            [
                "vehicle_id",
                "prefix_time_step",
                "remaining_time_steps",
                "event_observed",
                "status",
                "censor_survival_G",
                "ipcw_raw",
                "ipcw_normalized",
            ]
        ],
        on=["vehicle_id", "prefix_time_step"],
        how="inner",
        validate="one_to_one",
    )

    if len(calibration_fit) != len(calibration_pred):
        raise RuntimeError(
            "Not all labelled calibration predictions matched the IPCW audit. "
            f"Predictions={len(calibration_pred)}, "
            f"matched={len(calibration_fit)}."
        )

    if not np.array_equal(
        calibration_fit["label"].astype(int).to_numpy(),
        labelled_audit.set_index("vehicle_id")
        .loc[calibration_fit["vehicle_id"], "label"]
        .astype(int)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Calibration labels disagree between prediction and landmark tables."
        )

    calibration_fit.to_csv(
        predictions_dir / "calibration_ipcw_weighted_predictions.csv",
        index=False,
    )

    diagnostics = summarize_ipcw(
        full_landmark_audit=landmark_audit,
        weighted_labelled_landmarks=weighted_landmarks,
        km_table=km_table,
        tau=tau,
    )

    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in asdict(diagnostics).items()
        ]
    ).to_csv(
        tables_dir / "ipcw_diagnostics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Fit IPCW Platt to the frozen GRU raw logits.
    # ------------------------------------------------------------------
    calibration_cfg = cfg.get("calibration", {})
    platt_C = float(calibration_cfg.get("platt_C", 1e6))
    ece_bins = int(calibration_cfg.get("ece_bins", 10))
    reliability_bins = int(
        calibration_cfg.get("reliability_bins", 10)
    )

    ipcw_platt = fit_ipcw_platt_scaler(
        raw_logits=calibration_fit["raw_logit"],
        y_true=calibration_fit["label"],
        sample_weight=calibration_fit["ipcw_normalized"],
        C=platt_C,
        seed=seed,
    )
    joblib.dump(
        ipcw_platt,
        checkpoints_dir / "ipcw_platt_scaler_stage3_1.joblib",
    )

    ipcw_intercept = float(ipcw_platt.intercept_[0])
    ipcw_slope = float(ipcw_platt.coef_[0, 0])

    if ipcw_slope <= 0:
        print(
            "[WARNING] IPCW Platt slope is non-positive. "
            "Do not use this mapping operationally until investigated."
        )

    # ------------------------------------------------------------------
    # 4. Apply frozen IPCW calibrator to public validation predictions.
    # ------------------------------------------------------------------
    validation["ipcw_platt_probability"] = apply_platt_scaler(
        model=ipcw_platt,
        raw_logits=validation["raw_logit"],
    )

    validation.to_csv(
        predictions_dir
        / "validation_calibrated_predictions_stage3_1.csv",
        index=False,
    )

    y_val = validation["binary_label"].astype(int).to_numpy()

    methods = {
        "raw_weighted_sigmoid": validation[
            "raw_probability"
        ].to_numpy(dtype=float),
        "analytical_wbce_correction": validation[
            "analytical_probability"
        ].to_numpy(dtype=float),
        "heldout_platt": validation[
            "platt_probability"
        ].to_numpy(dtype=float),
        "ipcw_heldout_platt": validation[
            "ipcw_platt_probability"
        ].to_numpy(dtype=float),
    }

    metric_rows = []
    quantile_frames = []
    reliability_frames = []
    support_frames = []

    for method, probabilities in methods.items():
        metric_rows.append(
            {
                "method": method,
                **probability_metrics(
                    y_true=y_val,
                    probabilities=probabilities,
                    ece_bins=ece_bins,
                ),
            }
        )
        quantile_frames.append(
            probability_quantiles(
                probabilities,
                method=method,
            )
        )
        reliability_frames.append(
            reliability_table_quantile(
                y_true=y_val,
                probabilities=probabilities,
                method=method,
                n_bins=reliability_bins,
            )
        )
        support_frames.append(
            threshold_support_table(
                probabilities=probabilities,
                method=method,
            )
        )

    metrics_df = pd.DataFrame(metric_rows)
    quantiles_df = pd.concat(
        quantile_frames,
        ignore_index=True,
    )
    reliability_df = pd.concat(
        reliability_frames,
        ignore_index=True,
    )
    support_df = pd.concat(
        support_frames,
        ignore_index=True,
    )

    metrics_df.to_csv(
        tables_dir / "calibration_metrics_validation_stage3_1.csv",
        index=False,
    )
    quantiles_df.to_csv(
        tables_dir / "calibrated_probability_quantiles_stage3_1.csv",
        index=False,
    )
    reliability_df.to_csv(
        tables_dir / "calibration_reliability_bins_stage3_1.csv",
        index=False,
    )
    support_df.to_csv(
        tables_dir / "risk_threshold_support_stage3_1.csv",
        index=False,
    )

    parameter_table = pd.DataFrame(
        [
            {
                "method": "ipcw_heldout_platt",
                "parameter": "intercept",
                "value": ipcw_intercept,
            },
            {
                "method": "ipcw_heldout_platt",
                "parameter": "slope",
                "value": ipcw_slope,
            },
            {
                "method": "ipcw_heldout_platt",
                "parameter": "C",
                "value": platt_C,
            },
            {
                "method": "ipcw",
                "parameter": "G_tau_minus",
                "value": diagnostics.censor_survival_before_tau,
            },
            {
                "method": "ipcw",
                "parameter": "effective_sample_size",
                "value": diagnostics.effective_sample_size,
            },
            {
                "method": "ipcw",
                "parameter": "weighted_calibration_prevalence",
                "value": diagnostics.ipcw_weighted_positive_prevalence,
            },
        ]
    )
    parameter_table.to_csv(
        tables_dir / "ipcw_calibration_parameters.csv",
        index=False,
    )

    # Evaluation-only null Brier benchmark.
    validation_prevalence = float(np.mean(y_val))
    null_brier = float(
        np.mean((y_val - validation_prevalence) ** 2)
    )
    metrics_df["brier_skill_vs_validation_prevalence"] = (
        1.0 - metrics_df["brier_score"] / null_brier
    )
    metrics_df.to_csv(
        tables_dir / "calibration_metrics_validation_stage3_1.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. White-background reviewer-facing figures.
    # ------------------------------------------------------------------
    save_reliability_diagram(
        reliability_table=reliability_df,
        output_path=(
            figures_dir
            / "calibration_reliability_validation_stage3_1.png"
        ),
    )
    save_probability_histogram(
        probabilities=validation["ipcw_platt_probability"],
        output_path=(
            figures_dir
            / "probability_histogram_ipcw_heldout_platt.png"
        ),
        title=(
            "Validation probability distribution: "
            "IPCW held-out Platt"
        ),
    )

    # ------------------------------------------------------------------
    # 6. Summary.
    # ------------------------------------------------------------------
    summary = {
        "stage": "3.1",
        "gru_retrained": False,
        "validation_used_to_fit_calibrator": False,
        "tau": tau,
        "ipcw_diagnostics": asdict(diagnostics),
        "ipcw_platt": {
            "intercept": ipcw_intercept,
            "slope": ipcw_slope,
            "C": platt_C,
        },
        "validation_prevalence": validation_prevalence,
        "validation_null_brier": null_brier,
        "validation_metrics": metrics_df.to_dict(orient="records"),
        "interpretation_guardrail": (
            "IPCW corrects selective horizon-outcome observability induced by "
            "right censoring under the estimated censoring model. It does not "
            "guarantee correction of every distributional difference between "
            "the internal calibration landmarks and the public validation "
            "cohort."
        ),
    }

    with (logs_dir / "stage3_1_ipcw_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("-" * 78)
    print("STAGE 3.1 IPCW CALIBRATION RESULTS")
    print("-" * 78)
    print(
        f"Landmarks in censoring KM        : "
        f"{diagnostics.n_landmarks_for_censoring_model:,}"
    )
    print(
        f"Censoring endpoints              : "
        f"{diagnostics.n_censoring_events:,}"
    )
    print(
        f"Observed event endpoints         : "
        f"{diagnostics.n_observed_events:,}"
    )
    print(
        f"G(tau-)                          : "
        f"{diagnostics.censor_survival_before_tau:.6f}"
    )
    print(
        f"Unweighted calibration prevalence: "
        f"{diagnostics.unweighted_positive_prevalence:.6f}"
    )
    print(
        f"IPCW-weighted calibration prev.  : "
        f"{diagnostics.ipcw_weighted_positive_prevalence:.6f}"
    )
    print(
        f"IPCW effective sample size       : "
        f"{diagnostics.effective_sample_size:.2f}"
    )
    print(
        f"IPCW raw weight range            : "
        f"{diagnostics.raw_weight_min:.4f} "
        f"to {diagnostics.raw_weight_max:.4f}"
    )
    print()
    print(
        f"IPCW Platt intercept             : "
        f"{ipcw_intercept:.6f}"
    )
    print(
        f"IPCW Platt slope                 : "
        f"{ipcw_slope:.6f}"
    )
    print()

    display_columns = [
        "method",
        "auroc",
        "pr_auc",
        "enrichment",
        "brier_score",
        "brier_skill_vs_validation_prevalence",
        "log_loss",
        "ece_equal_width",
        "diagnostic_calibration_intercept",
        "diagnostic_calibration_slope",
        "mean_probability",
        "median_probability",
    ]
    print(
        metrics_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print()
    print(
        "IPCW fitting used only the held-out calibration landmarks and their "
        "censoring information. Public validation labels were used only for "
        "evaluation diagnostics."
    )
    print()
    print("Saved:")
    for path in [
        tables_dir / "ipcw_censoring_km.csv",
        tables_dir / "ipcw_diagnostics.csv",
        tables_dir / "ipcw_calibration_parameters.csv",
        tables_dir / "calibration_metrics_validation_stage3_1.csv",
        tables_dir / "calibrated_probability_quantiles_stage3_1.csv",
        tables_dir / "calibration_reliability_bins_stage3_1.csv",
        tables_dir / "risk_threshold_support_stage3_1.csv",
        predictions_dir / "calibration_ipcw_weighted_predictions.csv",
        predictions_dir
        / "validation_calibrated_predictions_stage3_1.csv",
        checkpoints_dir / "ipcw_platt_scaler_stage3_1.joblib",
        figures_dir
        / "calibration_reliability_validation_stage3_1.png",
        figures_dir
        / "probability_histogram_ipcw_heldout_platt.png",
        logs_dir / "stage3_1_ipcw_summary.json",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
