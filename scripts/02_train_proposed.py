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
from maestro_pdm.data.labels import frame_to_horizon_samples
from maestro_pdm.data.loading import (
    detect_id_column,
    read_specifications,
    read_train_tte,
    read_validation_labels,
    validate_dataset_directory,
)
from maestro_pdm.data.preprocessing import (
    load_operational_preprocessor,
    transform_static_specs,
)
from maestro_pdm.evaluation.landmarks import (
    select_validation_like_landmarks,
    summarize_landmarks,
)
from maestro_pdm.evaluation.metrics import binary_ranking_metrics
from maestro_pdm.models.temporal_gru import TemporalRiskGRU
from maestro_pdm.reproducibility import set_global_seed
from maestro_pdm.training.losses import compute_balanced_class_weights
from maestro_pdm.training.trainer import (
    fit_with_early_stopping,
    make_optimizer,
    predict_full_sequences,
    predict_loader,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2E.1: train the fixed-horizon GRU using multiple "
            "model-training prefixes, select the epoch using one "
            "validation-like landmark per tuning vehicle, and evaluate "
            "the frozen model on the full public validation cohort."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/paper_experiment.yaml",
    )
    return parser.parse_args()


def ensure_stage2a2b_outputs(output_root: Path):
    required = [
        output_root / "tables" / "development_split.csv",
        output_root / "tables" / "horizon_prefix_samples.csv",
        output_root / "checkpoints" / "operational_preprocessor.joblib",
        output_root / "checkpoints" / "static_onehot_encoder.joblib",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 2A+2B outputs are incomplete. Run the FULL audit first:\n"
            "python scripts/01_audit_dataset.py "
            "--config configs/paper_experiment.yaml\n\nMissing:\n  - "
            + "\n  - ".join(missing)
        )


def identify_validation_class_column(
    labels: pd.DataFrame,
    id_col: str,
) -> str:
    if "class_label" in labels.columns:
        return "class_label"

    remaining = [c for c in labels.columns if c != id_col]
    if len(remaining) == 1:
        return remaining[0]

    raise ValueError(
        "Could not identify validation class-label column."
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["experiment"]["seed"])
    tau = int(cfg["experiment"]["tau"])
    set_global_seed(seed)

    data_dir = get_data_dir()
    validate_dataset_directory(data_dir)
    output_root = get_output_dir(cfg)
    ensure_stage2a2b_outputs(output_root)

    tables_dir = output_root / "tables"
    predictions_dir = output_root / "predictions"
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    for p in (tables_dir, predictions_dir, checkpoints_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("STAGE 2E.1")
    print("Validation-like vehicle-level tuning landmark selection")
    print("=" * 78)
    print(f"Device               : {device}")
    print(f"Prediction horizon   : tau = {tau}")
    print(f"Landmark seed        : {seed}")
    print()

    # ------------------------------------------------------------------
    # Development split and original multi-prefix samples
    # ------------------------------------------------------------------
    split_df = pd.read_csv(
        tables_dir / "development_split.csv"
    )
    sample_df = pd.read_csv(
        tables_dir / "horizon_prefix_samples.csv"
    )
    all_samples = frame_to_horizon_samples(sample_df)

    model_train_ids = set(
        split_df.loc[
            split_df["split"] == "model_train",
            "vehicle_id",
        ].astype(int)
    )
    tuning_ids = set(
        split_df.loc[
            split_df["split"] == "tuning",
            "vehicle_id",
        ].astype(int)
    )

    preprocessor = load_operational_preprocessor(
        checkpoints_dir / "operational_preprocessor.joblib"
    )
    static_encoder = joblib.load(
        checkpoints_dir / "static_onehot_encoder.joblib"
    )

    train_specs = read_specifications(data_dir, "train")
    train_spec_id_col = detect_id_column(train_specs)
    spec_feature_cols = [
        c for c in train_specs.columns if c != train_spec_id_col
    ]
    train_static_vectors = transform_static_specs(
        encoder=static_encoder,
        specs_df=train_specs,
        id_col=train_spec_id_col,
        feature_cols=spec_feature_cols,
    )

    # Load operational histories for model-training + tuning vehicles.
    dev_ids = model_train_ids.union(tuning_ids)
    print(
        "Loading and preprocessing model-training/tuning sequences "
        "(streaming source CSV)..."
    )
    dev_sequences, _, _ = load_preprocessed_sequences(
        data_dir=data_dir,
        split="train",
        preprocessor=preprocessor,
        chunk_rows=int(cfg["data"]["ops_chunk_rows"]),
        allowed_vehicle_ids=dev_ids,
        max_vehicles=cfg["data"]["max_train_vehicles"],
    )

    # ------------------------------------------------------------------
    # TRAINING: retain multiple deterministic prefixes per vehicle.
    # ------------------------------------------------------------------
    train_dataset = HorizonPrefixDataset(
        sequences=dev_sequences,
        static_vectors=train_static_vectors,
        samples=all_samples,
        split="model_train",
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No model-training horizon samples are available."
        )

    # ------------------------------------------------------------------
    # TUNING: exactly one pseudo-validation landmark per vehicle.
    #
    # The landmark is chosen before inspecting Y_i(48). If the selected
    # prefix is censored before the complete 48-step horizon, the vehicle is
    # excluded rather than being resampled at a more convenient time.
    # ------------------------------------------------------------------
    tte = read_train_tte(data_dir)
    tte_id_col = detect_id_column(tte)

    tuning_samples, tuning_landmark_audit = (
        select_validation_like_landmarks(
            sequences=dev_sequences,
            vehicle_ids=tuning_ids,
            tte_df=tte,
            tte_id_col=tte_id_col,
            tau=tau,
            min_prefix_length=int(
                cfg["data"]["min_prefix_length"]
            ),
            seed=seed,
            split_name="tuning",
        )
    )

    tuning_landmark_audit.to_csv(
        tables_dir / "tuning_vehicle_landmarks.csv",
        index=False,
    )
    tuning_landmark_summary = summarize_landmarks(
        tuning_landmark_audit
    )
    tuning_landmark_summary.to_csv(
        tables_dir / "tuning_landmark_audit.csv",
        index=False,
    )

    tuning_dataset = HorizonPrefixDataset(
        sequences=dev_sequences,
        static_vectors=train_static_vectors,
        samples=tuning_samples,
        split="tuning",
    )

    if len(tuning_dataset) == 0:
        raise RuntimeError(
            "No labelled validation-like tuning landmarks are available."
        )

    # Exactly one labelled sample per included tuning vehicle.
    tuning_vehicle_count = len(
        {s.vehicle_id for s in tuning_dataset.samples}
    )
    if tuning_vehicle_count != len(tuning_dataset):
        raise RuntimeError(
            "The tuning landmark dataset contains duplicate vehicle IDs."
        )

    train_labels = np.asarray(
        [int(s.label) for s in train_dataset.samples],
        dtype=np.int64,
    )
    tuning_labels = np.asarray(
        [int(s.label) for s in tuning_dataset.samples],
        dtype=np.int64,
    )

    if len(np.unique(train_labels)) != 2:
        raise RuntimeError(
            "Model-training horizon samples do not contain both classes."
        )
    if len(np.unique(tuning_labels)) != 2:
        raise RuntimeError(
            "Vehicle-level tuning landmarks do not contain both classes. "
            "Do not change the seed based on validation performance; inspect "
            "the landmark audit before deciding on a pre-specified remedy."
        )

    # Guard against unstable PR-AUC selection from an extremely small number
    # of positives. This does not tune the seed; it simply fails visibly.
    tuning_positive_count = int((tuning_labels == 1).sum())
    if tuning_positive_count < 10:
        raise RuntimeError(
            "Fewer than 10 positive tuning landmarks were obtained "
            f"({tuning_positive_count}). Inspect "
            "outputs/tables/tuning_vehicle_landmarks.csv before proceeding."
        )

    class_weights = compute_balanced_class_weights(
        train_labels
    )

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(
        cfg["training"].get("num_workers", 0)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
        generator=generator,
    )
    tuning_loader = DataLoader(
        tuning_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
    )

    dynamic_dim = len(preprocessor.feature_names)
    static_dim = next(
        iter(train_static_vectors.values())
    ).shape[0]

    model = TemporalRiskGRU(
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        gru_hidden=int(cfg["model"]["gru_hidden"]),
        gru_layers=int(cfg["model"]["gru_layers"]),
        static_hidden=int(cfg["model"]["static_hidden"]),
        fusion_hidden=int(cfg["model"]["fusion_hidden"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = make_optimizer(model, cfg)

    landmark_row = tuning_landmark_summary.iloc[0]

    print(f"Dynamic input dimension p        : {dynamic_dim}")
    print(f"Encoded static dimension q_e     : {static_dim}")
    print(
        f"Model-training prefix samples    : "
        f"{len(train_dataset):,}"
    )
    print(
        f"Tuning candidate vehicles        : "
        f"{int(landmark_row['candidate_vehicles']):,}"
    )
    print(
        f"Labelled tuning landmarks        : "
        f"{len(tuning_dataset):,}"
    )
    print(
        f"Tuning landmark positives        : "
        f"{tuning_positive_count:,}"
    )
    print(
        f"Tuning landmark prevalence       : "
        f"{tuning_labels.mean():.6f}"
    )
    print(
        "Model-training class counts      : "
        f"N0={class_weights.n_negative:,}, "
        f"N1={class_weights.n_positive:,}"
    )
    print(
        "Weighted BCE class weights       : "
        f"omega_0={class_weights.omega_0:.6f}, "
        f"omega_1={class_weights.omega_1:.6f}"
    )
    print()

    # ------------------------------------------------------------------
    # Train and select epoch using VEHICLE-LEVEL tuning PR-AUC.
    # ------------------------------------------------------------------
    history, best_epoch, best_tune_pr_auc = (
        fit_with_early_stopping(
            model=model,
            train_loader=train_loader,
            tuning_loader=tuning_loader,
            optimizer=optimizer,
            device=device,
            omega_0=class_weights.omega_0,
            omega_1=class_weights.omega_1,
            epochs=int(cfg["training"]["epochs"]),
            patience=int(
                cfg["training"]["early_stopping_patience"]
            ),
            gradient_clip_norm=float(
                cfg["training"]["gradient_clip_norm"]
            ),
            checkpoint_path=(
                checkpoints_dir
                / "proposed_gru_best_vehicle_tuned.pt"
            ),
        )
    )
    history.to_csv(
        tables_dir
        / "proposed_training_history_vehicle_tuned.csv",
        index=False,
    )

    tuning_pred = predict_loader(
        model=model,
        loader=tuning_loader,
        device=device,
    )

    if tuning_pred["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Tuning prediction table contains duplicate vehicle IDs."
        )

    tuning_pred.to_csv(
        predictions_dir
        / "tuning_vehicle_landmark_raw_predictions.csv",
        index=False,
    )

    tuning_metrics = binary_ranking_metrics(
        tuning_pred["label"],
        tuning_pred["raw_probability"],
    )

    # ------------------------------------------------------------------
    # FINAL EVALUATION: complete public validation cohort.
    # No model selection is based on these results.
    # ------------------------------------------------------------------
    print()
    print("Loading complete public validation sequences...")

    validation_sequences, _, _ = (
        load_preprocessed_sequences(
            data_dir=data_dir,
            split="validation",
            preprocessor=preprocessor,
            chunk_rows=int(
                cfg["data"]["ops_chunk_rows"]
            ),
            allowed_vehicle_ids=None,
            max_vehicles=cfg["data"][
                "max_validation_vehicles"
            ],
        )
    )

    validation_specs = read_specifications(
        data_dir,
        "validation",
    )
    val_spec_id_col = detect_id_column(
        validation_specs
    )
    validation_static_vectors = (
        transform_static_specs(
            encoder=static_encoder,
            specs_df=validation_specs,
            id_col=val_spec_id_col,
            feature_cols=spec_feature_cols,
        )
    )

    validation_pred = predict_full_sequences(
        model=model,
        sequences=validation_sequences,
        static_vectors=validation_static_vectors,
        device=device,
        batch_size=batch_size,
    )

    validation_labels = read_validation_labels(
        data_dir
    )
    val_label_id_col = detect_id_column(
        validation_labels
    )
    val_class_col = identify_validation_class_column(
        validation_labels,
        val_label_id_col,
    )

    label_frame = validation_labels[
        [val_label_id_col, val_class_col]
    ].copy()
    label_frame = label_frame.rename(
        columns={
            val_label_id_col: "vehicle_id",
            val_class_col: "class_label",
        }
    )
    label_frame["vehicle_id"] = (
        label_frame["vehicle_id"].astype(int)
    )
    label_frame["class_label"] = (
        label_frame["class_label"].astype(int)
    )
    label_frame["binary_label"] = (
        label_frame["class_label"] > 0
    ).astype(int)

    validation_pred = label_frame.merge(
        validation_pred,
        on="vehicle_id",
        how="left",
        validate="one_to_one",
    )

    missing_predictions = int(
        validation_pred[
            "raw_probability"
        ].isna().sum()
    )
    if missing_predictions:
        missing_ids = validation_pred.loc[
            validation_pred[
                "raw_probability"
            ].isna(),
            "vehicle_id",
        ].tolist()[:20]
        raise RuntimeError(
            f"Missing predictions for "
            f"{missing_predictions} validation "
            f"vehicles. Example IDs: {missing_ids}"
        )

    expected_n = len(validation_labels)
    if len(validation_pred) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} validation rows "
            f"but obtained {len(validation_pred)}."
        )

    validation_pred.to_csv(
        predictions_dir
        / "validation_raw_predictions_vehicle_tuned.csv",
        index=False,
    )

    val_metrics = binary_ranking_metrics(
        validation_pred["binary_label"],
        validation_pred["raw_probability"],
    )

    metrics_table = pd.DataFrame(
        [
            {
                "dataset": "tuning_vehicle_landmarks",
                **tuning_metrics,
            },
            {
                "dataset": (
                    "public_validation_full_sequence"
                ),
                **val_metrics,
            },
        ]
    )
    metrics_table.to_csv(
        tables_dir
        / "proposed_raw_metrics_vehicle_tuned.csv",
        index=False,
    )

    model_config = {
        "experiment_name": cfg["experiment"]["name"],
        "stage": "2E.1",
        "seed": seed,
        "tau": tau,
        "dynamic_dimension_p": int(dynamic_dim),
        "static_encoding": cfg["preprocessing"][
            "static_encoding"
        ],
        "encoded_static_dimension_q_e": int(
            static_dim
        ),
        "gru_hidden_dimension_d_T": int(
            cfg["model"]["gru_hidden"]
        ),
        "gru_layers": int(
            cfg["model"]["gru_layers"]
        ),
        "static_hidden": int(
            cfg["model"]["static_hidden"]
        ),
        "fusion_hidden": int(
            cfg["model"]["fusion_hidden"]
        ),
        "dropout": float(cfg["model"]["dropout"]),
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate": float(
            cfg["training"]["learning_rate"]
        ),
        "weight_decay_lambda_reg": float(
            cfg["training"]["weight_decay"]
        ),
        "batch_size": batch_size,
        "maximum_epochs": int(
            cfg["training"]["epochs"]
        ),
        "early_stopping_patience": int(
            cfg["training"][
                "early_stopping_patience"
            ]
        ),
        "best_epoch": int(best_epoch),
        "gradient_clip_norm": float(
            cfg["training"]["gradient_clip_norm"]
        ),
        "omega_0": float(
            class_weights.omega_0
        ),
        "omega_1": float(
            class_weights.omega_1
        ),
        "model_train_N0": int(
            class_weights.n_negative
        ),
        "model_train_N1": int(
            class_weights.n_positive
        ),
        "activation": "ReLU",
        "output_activation": "sigmoid",
        "output_dimension": 1,
        "training_sampling": (
            "multiple deterministic prefixes per "
            "model-training vehicle"
        ),
        "model_selection_sampling": (
            "one seed-42 random readout landmark "
            "per tuning vehicle, selected before "
            "label inspection"
        ),
        "model_selection_metric": (
            "vehicle-level tuning PR-AUC"
        ),
        "calibration_applied": False,
    }

    pd.DataFrame(
        [
            {"parameter": k, "value": v}
            for k, v in model_config.items()
        ]
    ).to_csv(
        tables_dir
        / "proposed_model_configuration_vehicle_tuned.csv",
        index=False,
    )

    summary = {
        "stage": "2E.1",
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_tuning_pr_auc": float(
            best_tune_pr_auc
        ),
        "tuning_landmarks": {
            key: (
                None
                if pd.isna(value)
                else (
                    int(value)
                    if isinstance(
                        value,
                        (np.integer, int),
                    )
                    else float(value)
                    if isinstance(
                        value,
                        (np.floating, float),
                    )
                    else value
                )
            )
            for key, value
            in landmark_row.to_dict().items()
        },
        "validation": val_metrics,
        "calibration_status": (
            "not yet applied; the calibration "
            "split remains untouched"
        ),
    }

    with (
        logs_dir / "stage2e1_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("-" * 78)
    print("STAGE 2E.1 RESULTS")
    print("-" * 78)
    print(
        f"Best training epoch             : "
        f"{best_epoch}"
    )
    print(
        f"Best vehicle-level tuning PR-AUC: "
        f"{best_tune_pr_auc:.6f}"
    )
    print(
        f"Tuning labelled vehicles        : "
        f"{tuning_metrics['n']:,}"
    )
    print(
        f"Tuning positives                : "
        f"{tuning_metrics['n_positive']:,}"
    )
    print(
        f"Tuning positive prevalence      : "
        f"{tuning_metrics['positive_prevalence']:.6f}"
    )
    print(
        f"Tuning enrichment               : "
        f"{tuning_metrics['enrichment']:.6f}"
    )
    print()
    print(
        f"Validation vehicles             : "
        f"{val_metrics['n']:,}"
    )
    print(
        f"Validation positives            : "
        f"{val_metrics['n_positive']:,}"
    )
    print(
        f"Validation positive prevalence  : "
        f"{val_metrics['positive_prevalence']:.6f}"
    )
    print(
        f"Raw validation AUROC            : "
        f"{val_metrics['auroc']:.6f}"
    )
    print(
        f"Raw validation PR-AUC           : "
        f"{val_metrics['pr_auc']:.6f}"
    )
    print(
        f"Raw validation enrichment       : "
        f"{val_metrics['enrichment']:.6f}"
    )
    print()
    print(
        "The public validation metrics are reported "
        "for evaluation only; they were not used to "
        "select the epoch."
    )
    print()
    print("Saved:")
    for path in [
        tables_dir / "tuning_vehicle_landmarks.csv",
        tables_dir / "tuning_landmark_audit.csv",
        tables_dir
        / "proposed_training_history_vehicle_tuned.csv",
        tables_dir
        / "proposed_raw_metrics_vehicle_tuned.csv",
        tables_dir
        / "proposed_model_configuration_vehicle_tuned.csv",
        predictions_dir
        / "tuning_vehicle_landmark_raw_predictions.csv",
        predictions_dir
        / "validation_raw_predictions_vehicle_tuned.csv",
        checkpoints_dir
        / "proposed_gru_best_vehicle_tuned.pt",
        logs_dir / "stage2e1_summary.json",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
