from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from maestro_pdm.config import get_data_dir, get_output_dir, load_config
from maestro_pdm.data.datasets import (
    HorizonPrefixDataset,
    collate_variable_length,
    load_preprocessed_sequences,
)
from maestro_pdm.data.loading import (
    detect_id_column,
    read_specifications,
    read_train_tte,
    validate_dataset_directory,
)
from maestro_pdm.data.preprocessing import (
    load_operational_preprocessor,
    transform_static_specs,
)
from maestro_pdm.evaluation.calibration import (
    analytical_weight_correction_from_logits,
    apply_platt_scaler,
    fit_platt_scaler,
    platt_parameters,
    probability_metrics,
    probability_quantiles,
    reliability_table_quantile,
    threshold_support_table,
)
from maestro_pdm.evaluation.landmarks import (
    select_validation_like_landmarks,
    summarize_landmarks,
)
from maestro_pdm.models.temporal_gru import TemporalRiskGRU
from maestro_pdm.reproducibility import set_global_seed
from maestro_pdm.training.trainer import predict_loader
from maestro_pdm.visualization.calibration import (
    save_probability_histogram,
    save_reliability_diagram,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: calibrate the frozen Stage 2E.1 GRU using the untouched "
            "calibration vehicle split and compare raw, analytical WBCE "
            "correction, and held-out Platt probabilities."
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
            "Required frozen-model/Stage 2 outputs are missing:\n  - "
            + "\n  - ".join(missing)
        )


def configuration_dict(path: Path) -> dict:
    table = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(table.columns):
        raise ValueError(
            f"{path} must contain 'parameter' and 'value' columns."
        )
    return dict(zip(table["parameter"].astype(str), table["value"]))


def load_frozen_model(
    checkpoint_path: Path,
    cfg: dict,
    dynamic_dim: int,
    static_dim: int,
    device: torch.device,
):
    model = TemporalRiskGRU(
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        gru_hidden=int(cfg["model"]["gru_hidden"]),
        gru_layers=int(cfg["model"]["gru_layers"]),
        static_hidden=int(cfg["model"]["static_hidden"]),
        fusion_hidden=int(cfg["model"]["fusion_hidden"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["experiment"]["seed"])
    tau = int(cfg["experiment"]["tau"])
    set_global_seed(seed)

    data_dir = get_data_dir()
    validate_dataset_directory(data_dir)
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

    split_path = tables_dir / "development_split.csv"
    operational_preprocessor_path = (
        checkpoints_dir / "operational_preprocessor.joblib"
    )
    static_encoder_path = (
        checkpoints_dir / "static_onehot_encoder.joblib"
    )
    frozen_checkpoint_path = (
        checkpoints_dir / "proposed_gru_best_vehicle_tuned.pt"
    )
    frozen_config_path = (
        tables_dir / "proposed_model_configuration_vehicle_tuned.csv"
    )
    validation_raw_path = (
        predictions_dir
        / "validation_raw_predictions_vehicle_tuned.csv"
    )

    require_files(
        [
            split_path,
            operational_preprocessor_path,
            static_encoder_path,
            frozen_checkpoint_path,
            frozen_config_path,
            validation_raw_path,
        ]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("STAGE 3")
    print("Probability calibration of the frozen Stage 2E.1 model")
    print("=" * 78)
    print(f"Device               : {device}")
    print(f"Prediction horizon   : tau = {tau}")
    print("Model retraining     : NO")
    print()

    split_df = pd.read_csv(split_path)
    calibration_ids = set(
        split_df.loc[
            split_df["split"] == "calibration",
            "vehicle_id",
        ].astype(int)
    )

    if not calibration_ids:
        raise RuntimeError(
            "No vehicles were found in the calibration split."
        )

    preprocessor = load_operational_preprocessor(
        operational_preprocessor_path
    )
    static_encoder = joblib.load(static_encoder_path)

    train_specs = read_specifications(data_dir, "train")
    spec_id_col = detect_id_column(train_specs)
    spec_feature_cols = [
        c for c in train_specs.columns if c != spec_id_col
    ]
    static_vectors = transform_static_specs(
        encoder=static_encoder,
        specs_df=train_specs,
        id_col=spec_id_col,
        feature_cols=spec_feature_cols,
    )

    dynamic_dim = len(preprocessor.feature_names)
    static_dim = next(iter(static_vectors.values())).shape[0]

    frozen_cfg = configuration_dict(frozen_config_path)
    omega_0 = float(frozen_cfg["omega_0"])
    omega_1 = float(frozen_cfg["omega_1"])

    model, checkpoint = load_frozen_model(
        checkpoint_path=frozen_checkpoint_path,
        cfg=cfg,
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        device=device,
    )

    print(
        f"Frozen best epoch    : "
        f"{checkpoint.get('best_epoch', 'unknown')}"
    )
    print(f"omega_0              : {omega_0:.6f}")
    print(f"omega_1              : {omega_1:.6f}")
    print()

    # ------------------------------------------------------------------
    # Calibration split: one validation-like landmark per vehicle.
    # ------------------------------------------------------------------
    print(
        "Loading calibration-vehicle sequences and selecting one "
        "validation-like landmark per vehicle..."
    )
    calibration_sequences, _, _ = load_preprocessed_sequences(
        data_dir=data_dir,
        split="train",
        preprocessor=preprocessor,
        chunk_rows=int(cfg["data"]["ops_chunk_rows"]),
        allowed_vehicle_ids=calibration_ids,
        max_vehicles=None,
    )

    tte = read_train_tte(data_dir)
    tte_id_col = detect_id_column(tte)

    calibration_samples, calibration_landmark_audit = (
        select_validation_like_landmarks(
            sequences=calibration_sequences,
            vehicle_ids=calibration_ids,
            tte_df=tte,
            tte_id_col=tte_id_col,
            tau=tau,
            min_prefix_length=int(
                cfg["data"]["min_prefix_length"]
            ),
            seed=seed,
            split_name="calibration",
        )
    )

    calibration_landmark_audit.to_csv(
        tables_dir / "calibration_vehicle_landmarks.csv",
        index=False,
    )
    calibration_landmark_summary = summarize_landmarks(
        calibration_landmark_audit
    )
    calibration_landmark_summary.to_csv(
        tables_dir / "calibration_landmark_audit.csv",
        index=False,
    )

    calibration_dataset = HorizonPrefixDataset(
        sequences=calibration_sequences,
        static_vectors=static_vectors,
        samples=calibration_samples,
        split="calibration",
    )

    if len(calibration_dataset) == 0:
        raise RuntimeError(
            "No labelled calibration landmarks are available."
        )

    calibration_positive_count = int(
        sum(int(s.label) for s in calibration_dataset.samples)
    )
    if calibration_positive_count < 10:
        raise RuntimeError(
            "Fewer than 10 positive calibration landmarks were obtained. "
            "Inspect calibration_vehicle_landmarks.csv before proceeding."
        )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(
            cfg["training"].get("num_workers", 0)
        ),
        collate_fn=collate_variable_length,
    )

    calibration_pred = predict_loader(
        model=model,
        loader=calibration_loader,
        device=device,
    )

    if calibration_pred["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Calibration predictions contain duplicate vehicle IDs."
        )

    calibration_pred.to_csv(
        predictions_dir / "calibration_raw_predictions.csv",
        index=False,
    )

    calibration_prevalence = float(
        calibration_pred["label"].mean()
    )

    # ------------------------------------------------------------------
    # Fit held-out Platt scaler to FROZEN raw logits.
    # ------------------------------------------------------------------
    calibration_cfg = cfg.get("calibration", {})
    platt_C = float(calibration_cfg.get("platt_C", 1e6))
    ece_bins = int(calibration_cfg.get("ece_bins", 10))
    reliability_bins = int(
        calibration_cfg.get("reliability_bins", 10)
    )

    platt_model = fit_platt_scaler(
        raw_logits=calibration_pred["raw_logit"],
        y_true=calibration_pred["label"],
        C=platt_C,
        seed=seed,
    )
    joblib.dump(
        platt_model,
        checkpoints_dir / "platt_scaler_stage3.joblib",
    )
    platt = platt_parameters(platt_model)

    if platt.slope <= 0:
        print(
            "[WARNING] Fitted Platt slope is non-positive. "
            "This would reverse or flatten the risk ranking and requires "
            "methodological review before using Platt probabilities."
        )

    # ------------------------------------------------------------------
    # Validation predictions are NEVER used to fit calibration.
    # They are read only after the calibrator is frozen.
    # ------------------------------------------------------------------
    validation = pd.read_csv(validation_raw_path)

    required_validation_columns = {
        "vehicle_id",
        "binary_label",
        "raw_logit",
        "raw_probability",
    }
    missing = required_validation_columns.difference(validation.columns)
    if missing:
        raise ValueError(
            "Validation raw-prediction table is missing columns: "
            f"{sorted(missing)}"
        )

    if validation["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Validation raw predictions contain duplicate vehicle IDs."
        )
    if validation[list(required_validation_columns)].isna().any().any():
        raise RuntimeError(
            "Validation raw predictions contain missing values."
        )

    y_val = validation["binary_label"].astype(int).to_numpy()
    raw_logits = validation["raw_logit"].astype(float).to_numpy()
    raw_probability = validation["raw_probability"].astype(float).to_numpy()

    validation["analytical_probability"] = (
        analytical_weight_correction_from_logits(
            raw_logits=raw_logits,
            omega_0=omega_0,
            omega_1=omega_1,
        )
    )
    validation["platt_probability"] = apply_platt_scaler(
        model=platt_model,
        raw_logits=raw_logits,
    )

    validation.to_csv(
        predictions_dir / "validation_calibrated_predictions.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Calibration/ranking metrics
    # ------------------------------------------------------------------
    methods = {
        "raw_weighted_sigmoid": raw_probability,
        "analytical_wbce_correction": validation[
            "analytical_probability"
        ].to_numpy(),
        "heldout_platt": validation[
            "platt_probability"
        ].to_numpy(),
    }

    metric_rows = []
    quantile_frames = []
    reliability_frames = []
    support_frames = []

    for method, probabilities in methods.items():
        metrics = probability_metrics(
            y_true=y_val,
            probabilities=probabilities,
            ece_bins=ece_bins,
        )
        metric_rows.append(
            {
                "method": method,
                **metrics,
            }
        )
        quantile_frames.append(
            probability_quantiles(
                probabilities=probabilities,
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
        tables_dir / "calibration_metrics_validation.csv",
        index=False,
    )
    quantiles_df.to_csv(
        tables_dir / "calibrated_probability_quantiles.csv",
        index=False,
    )
    reliability_df.to_csv(
        tables_dir / "calibration_reliability_bins.csv",
        index=False,
    )
    support_df.to_csv(
        tables_dir / "risk_threshold_support.csv",
        index=False,
    )

    calibration_parameters_df = pd.DataFrame(
        [
            {
                "method": "analytical_wbce_correction",
                "parameter": "omega_0",
                "value": omega_0,
            },
            {
                "method": "analytical_wbce_correction",
                "parameter": "omega_1",
                "value": omega_1,
            },
            {
                "method": "analytical_wbce_correction",
                "parameter": "logit_shift_log_omega0_over_omega1",
                "value": float(np.log(omega_0 / omega_1)),
            },
            {
                "method": "heldout_platt",
                "parameter": "intercept",
                "value": platt.intercept,
            },
            {
                "method": "heldout_platt",
                "parameter": "slope",
                "value": platt.slope,
            },
            {
                "method": "heldout_platt",
                "parameter": "C",
                "value": platt.C,
            },
        ]
    )
    calibration_parameters_df.to_csv(
        tables_dir / "calibration_parameters.csv",
        index=False,
    )

    # Prevalence audit: the mismatch is reported explicitly, not hidden.
    tuning_audit_path = tables_dir / "tuning_landmark_audit.csv"
    tuning_prevalence = np.nan
    tuning_n = np.nan
    if tuning_audit_path.exists():
        tuning_audit = pd.read_csv(tuning_audit_path)
        if len(tuning_audit):
            tuning_prevalence = float(
                tuning_audit.loc[0, "positive_prevalence"]
            )
            tuning_n = int(
                tuning_audit.loc[
                    0,
                    "included_labelled_vehicles",
                ]
            )

    prevalence_comparison = pd.DataFrame(
        [
            {
                "population": "tuning_vehicle_landmarks",
                "n": tuning_n,
                "positive_prevalence": tuning_prevalence,
                "role": "model selection only",
            },
            {
                "population": "calibration_vehicle_landmarks",
                "n": int(len(calibration_pred)),
                "positive_prevalence": calibration_prevalence,
                "role": "Platt fitting only",
            },
            {
                "population": "public_validation",
                "n": int(len(validation)),
                "positive_prevalence": float(np.mean(y_val)),
                "role": "final evaluation only",
            },
        ]
    )
    prevalence_comparison.to_csv(
        tables_dir / "prevalence_comparison.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figures: white background, no grid.
    # ------------------------------------------------------------------
    save_reliability_diagram(
        reliability_table=reliability_df,
        output_path=figures_dir / "calibration_reliability_validation.png",
    )

    for method, probabilities in methods.items():
        safe_name = method.replace(" ", "_")
        save_probability_histogram(
            probabilities=probabilities,
            output_path=(
                figures_dir
                / f"probability_histogram_{safe_name}.png"
            ),
            title=f"Validation probability distribution: {method}",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "stage": "3",
        "model_retrained": False,
        "frozen_checkpoint": str(frozen_checkpoint_path),
        "calibration_landmarks": {
            "n": int(len(calibration_pred)),
            "n_positive": calibration_positive_count,
            "positive_prevalence": calibration_prevalence,
        },
        "validation": {
            "n": int(len(validation)),
            "n_positive": int(y_val.sum()),
            "positive_prevalence": float(y_val.mean()),
        },
        "analytical_correction": {
            "omega_0": omega_0,
            "omega_1": omega_1,
            "logit_shift": float(np.log(omega_0 / omega_1)),
        },
        "platt": {
            "intercept": platt.intercept,
            "slope": platt.slope,
            "C": platt.C,
            "fit_population": "calibration_vehicle_landmarks",
        },
        "validation_metrics": metrics_df.to_dict(orient="records"),
        "important_note": (
            "Validation labels were used only for evaluation diagnostics; "
            "neither the GRU nor either calibration mapping was fit or selected "
            "using public validation labels."
        ),
    }

    with (logs_dir / "stage3_calibration_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("-" * 78)
    print("STAGE 3 CALIBRATION RESULTS")
    print("-" * 78)
    print(
        f"Calibration labelled vehicles    : {len(calibration_pred):,}"
    )
    print(
        f"Calibration positives            : {calibration_positive_count:,}"
    )
    print(
        f"Calibration positive prevalence  : {calibration_prevalence:.6f}"
    )
    print(
        f"Validation positive prevalence   : {float(y_val.mean()):.6f}"
    )
    print()
    print(
        f"Platt intercept                  : {platt.intercept:.6f}"
    )
    print(
        f"Platt slope                      : {platt.slope:.6f}"
    )
    print()

    display_columns = [
        "method",
        "auroc",
        "pr_auc",
        "enrichment",
        "brier_score",
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
        "The calibration mappings were fitted without public validation "
        "labels. Validation labels are used only to report calibration "
        "diagnostics."
    )
    print()
    print("Saved:")
    for path in [
        tables_dir / "calibration_landmark_audit.csv",
        tables_dir / "calibration_vehicle_landmarks.csv",
        tables_dir / "calibration_metrics_validation.csv",
        tables_dir / "calibration_parameters.csv",
        tables_dir / "calibrated_probability_quantiles.csv",
        tables_dir / "calibration_reliability_bins.csv",
        tables_dir / "risk_threshold_support.csv",
        tables_dir / "prevalence_comparison.csv",
        predictions_dir / "calibration_raw_predictions.csv",
        predictions_dir / "validation_calibrated_predictions.csv",
        checkpoints_dir / "platt_scaler_stage3.joblib",
        figures_dir / "calibration_reliability_validation.png",
        logs_dir / "stage3_calibration_summary.json",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
