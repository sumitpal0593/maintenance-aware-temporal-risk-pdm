from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from maestro_pdm.config import get_data_dir, get_output_dir, load_config
from maestro_pdm.data.datasets import load_preprocessed_sequences
from maestro_pdm.data.labels import frame_to_horizon_samples
from maestro_pdm.data.loading import (
    detect_id_column,
    read_specifications,
    validate_dataset_directory,
)
from maestro_pdm.data.preprocessing import (
    load_operational_preprocessor,
    transform_static_specs,
)
from maestro_pdm.evaluation.comparisons import (
    representation_comparison_summary,
)
from maestro_pdm.evaluation.metrics import binary_ranking_metrics
from maestro_pdm.features.representations import (
    aggregate_feature_names,
    build_full_sequence_representation,
    build_prefix_representation,
)
from maestro_pdm.models.controls import (
    extract_gru_full_embeddings,
    extract_gru_prefix_embeddings,
    fit_logistic_control,
    save_control_model,
)
from maestro_pdm.models.temporal_gru import TemporalRiskGRU
from maestro_pdm.reproducibility import set_global_seed


CONTROL_C_GRID = [0.01, 0.1, 1.0, 10.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4A: reviewer-requested temporal-encoding controls. "
            "Compare hand-engineered aggregate summaries and frozen GRU "
            "representations using the same logistic-regression classifier."
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
            "Required frozen-model/Stage 2E.1 outputs are missing:\n  - "
            + "\n  - ".join(missing)
        )


def load_frozen_gru(
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

    model.load_state_dict(
        checkpoint.get("state_dict", checkpoint)
    )
    model.eval()
    return model, checkpoint


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed)

    data_dir = get_data_dir()
    validate_dataset_directory(data_dir)
    output_root = get_output_dir(cfg)

    tables_dir = output_root / "tables"
    predictions_dir = output_root / "predictions"
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    features_dir = output_root / "features"

    for p in (
        tables_dir,
        predictions_dir,
        checkpoints_dir,
        logs_dir,
        features_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    required = [
        tables_dir / "development_split.csv",
        tables_dir / "horizon_prefix_samples.csv",
        tables_dir / "tuning_vehicle_landmarks.csv",
        checkpoints_dir / "operational_preprocessor.joblib",
        checkpoints_dir / "static_onehot_encoder.joblib",
        checkpoints_dir / "proposed_gru_best_vehicle_tuned.pt",
        predictions_dir
        / "validation_raw_predictions_vehicle_tuned.csv",
    ]
    require_files(required)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 78)
    print("STAGE 4A")
    print("Reviewer-requested temporal-encoding control experiment")
    print("=" * 78)
    print(f"Device                    : {device}")
    print("Classifier for controls   : L2 logistic regression")
    print(f"C grid                    : {CONTROL_C_GRID}")
    print("Validation used for tuning: NO")
    print()

    split_df = pd.read_csv(
        tables_dir / "development_split.csv"
    )
    all_sample_df = pd.read_csv(
        tables_dir / "horizon_prefix_samples.csv"
    )
    all_samples = frame_to_horizon_samples(all_sample_df)

    tuning_landmark_df = pd.read_csv(
        tables_dir / "tuning_vehicle_landmarks.csv"
    )
    tuning_samples = frame_to_horizon_samples(
        tuning_landmark_df[
            tuning_landmark_df["label"].notna()
        ].copy()
    )

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
    static_feature_cols = [
        c for c in train_specs.columns
        if c != train_spec_id_col
    ]
    train_static = transform_static_specs(
        encoder=static_encoder,
        specs_df=train_specs,
        id_col=train_spec_id_col,
        feature_cols=static_feature_cols,
    )

    dev_ids = model_train_ids.union(tuning_ids)

    print("Loading model-training and tuning sequences...")
    dev_sequences, _, _ = load_preprocessed_sequences(
        data_dir=data_dir,
        split="train",
        preprocessor=preprocessor,
        chunk_rows=int(cfg["data"]["ops_chunk_rows"]),
        allowed_vehicle_ids=dev_ids,
        max_vehicles=None,
    )

    model_train_samples = [
        s
        for s in all_samples
        if s.split == "model_train"
        and s.label is not None
        and s.vehicle_id in model_train_ids
    ]

    print("Building conventional aggregate representations...")
    (
        train_meta,
        X_train_agg,
        X_train_static,
        X_train_time,
    ) = build_prefix_representation(
        samples=model_train_samples,
        sequences=dev_sequences,
        static_vectors=train_static,
    )

    (
        tune_meta,
        X_tune_agg,
        X_tune_static,
        X_tune_time,
    ) = build_prefix_representation(
        samples=tuning_samples,
        sequences=dev_sequences,
        static_vectors=train_static,
    )

    # Ensure the tuning table is genuinely one row per vehicle.
    if tune_meta["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Vehicle-level tuning representation contains duplicate vehicles."
        )

    y_train = train_meta["label"].to_numpy(dtype=int)
    y_tune = tune_meta["label"].to_numpy(dtype=int)

    dynamic_dim = len(preprocessor.feature_names)
    static_dim = X_train_static.shape[1]

    frozen_gru, frozen_checkpoint = load_frozen_gru(
        checkpoint_path=(
            checkpoints_dir / "proposed_gru_best_vehicle_tuned.pt"
        ),
        cfg=cfg,
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        device=device,
    )

    print("Extracting frozen GRU temporal representations h_i...")
    gru_train_meta, X_train_gru = extract_gru_prefix_embeddings(
        model=frozen_gru,
        sequences=dev_sequences,
        static_vectors=train_static,
        samples=model_train_samples,
        split="model_train",
        device=device,
        batch_size=int(cfg["training"]["batch_size"]),
    )
    gru_tune_meta, X_tune_gru = extract_gru_prefix_embeddings(
        model=frozen_gru,
        sequences=dev_sequences,
        static_vectors=train_static,
        samples=tuning_samples,
        split="tuning",
        device=device,
        batch_size=int(cfg["training"]["batch_size"]),
    )

    # Alignment checks are explicit because the reviewer control is only valid
    # if the same samples and labels are compared across representations.
    train_key = list(
        zip(
            train_meta["vehicle_id"],
            train_meta["prefix_time_step"],
            train_meta["label"],
        )
    )
    gru_train_key = list(
        zip(
            gru_train_meta["vehicle_id"],
            gru_train_meta["prefix_time_step"],
            gru_train_meta["label"],
        )
    )
    tune_key = list(
        zip(
            tune_meta["vehicle_id"],
            tune_meta["prefix_time_step"],
            tune_meta["label"],
        )
    )
    gru_tune_key = list(
        zip(
            gru_tune_meta["vehicle_id"],
            gru_tune_meta["prefix_time_step"],
            gru_tune_meta["label"],
        )
    )

    if train_key != gru_train_key:
        raise RuntimeError(
            "Aggregate and GRU training representations are misaligned."
        )
    if tune_key != gru_tune_key:
        raise RuntimeError(
            "Aggregate and GRU tuning representations are misaligned."
        )

    # ------------------------------------------------------------------
    # Validation: one full available history per public validation vehicle.
    # ------------------------------------------------------------------
    print("Loading complete public validation sequences...")
    val_sequences, _, _ = load_preprocessed_sequences(
        data_dir=data_dir,
        split="validation",
        preprocessor=preprocessor,
        chunk_rows=int(cfg["data"]["ops_chunk_rows"]),
        allowed_vehicle_ids=None,
        max_vehicles=cfg["data"]["max_validation_vehicles"],
    )

    val_specs = read_specifications(data_dir, "validation")
    val_spec_id_col = detect_id_column(val_specs)
    val_static = transform_static_specs(
        encoder=static_encoder,
        specs_df=val_specs,
        id_col=val_spec_id_col,
        feature_cols=static_feature_cols,
    )

    proposed_validation = pd.read_csv(
        predictions_dir
        / "validation_raw_predictions_vehicle_tuned.csv"
    )
    if proposed_validation["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Frozen proposed-model validation predictions contain "
            "duplicate vehicle IDs."
        )

    validation_ids = proposed_validation[
        "vehicle_id"
    ].astype(int).tolist()

    (
        val_meta,
        X_val_agg,
        X_val_static,
        X_val_time,
    ) = build_full_sequence_representation(
        sequences=val_sequences,
        static_vectors=val_static,
        vehicle_ids=validation_ids,
    )

    gru_val_meta, X_val_gru = extract_gru_full_embeddings(
        model=frozen_gru,
        sequences=val_sequences,
        vehicle_ids=validation_ids,
        device=device,
        batch_size=int(cfg["training"]["batch_size"]),
    )

    if not np.array_equal(
        val_meta["vehicle_id"].to_numpy(),
        gru_val_meta["vehicle_id"].to_numpy(),
    ):
        raise RuntimeError(
            "Aggregate and GRU validation representations are misaligned."
        )

    label_lookup = proposed_validation.set_index(
        "vehicle_id"
    )["binary_label"]
    y_val = (
        val_meta["vehicle_id"]
        .map(label_lookup)
        .astype(int)
        .to_numpy()
    )

    if len(y_val) != 5046:
        print(
            f"[WARNING] Expected 5,046 validation vehicles; got {len(y_val):,}."
        )

    # ------------------------------------------------------------------
    # Same classifier on multiple representations.
    # ------------------------------------------------------------------
    representations = {
        "time_only_logistic": (
            X_train_time,
            X_tune_time,
            X_val_time,
        ),
        "static_only_logistic": (
            X_train_static,
            X_tune_static,
            X_val_static,
        ),
        "aggregate_dynamic_logistic": (
            X_train_agg,
            X_tune_agg,
            X_val_agg,
        ),
        "gru_temporal_logistic": (
            X_train_gru,
            X_tune_gru,
            X_val_gru,
        ),
        "aggregate_dynamic_plus_static_logistic": (
            np.concatenate(
                [X_train_agg, X_train_static],
                axis=1,
            ),
            np.concatenate(
                [X_tune_agg, X_tune_static],
                axis=1,
            ),
            np.concatenate(
                [X_val_agg, X_val_static],
                axis=1,
            ),
        ),
        "gru_temporal_plus_static_logistic": (
            np.concatenate(
                [X_train_gru, X_train_static],
                axis=1,
            ),
            np.concatenate(
                [X_tune_gru, X_tune_static],
                axis=1,
            ),
            np.concatenate(
                [X_val_gru, X_val_static],
                axis=1,
            ),
        ),
    }

    tuning_tables = []
    metric_rows = []
    prediction_table = val_meta[
        ["vehicle_id", "prefix_time_step", "prefix_length"]
    ].copy()
    prediction_table["binary_label"] = y_val

    selected_hyperparameters = []

    for name, (Xtr, Xtu, Xv) in representations.items():
        print(
            f"Fitting {name} "
            f"(dimension={Xtr.shape[1]})..."
        )
        model, tuning_table, best_C = fit_logistic_control(
            X_train=Xtr,
            y_train=y_train,
            X_tuning=Xtu,
            y_tuning=y_tune,
            C_grid=CONTROL_C_GRID,
            seed=seed,
        )

        tuning_table.insert(0, "model", name)
        tuning_tables.append(tuning_table)

        p_val = model.predict_proba(Xv)[:, 1]
        metrics = binary_ranking_metrics(y_val, p_val)

        metric_rows.append(
            {
                "model": name,
                "representation_dimension": int(Xtr.shape[1]),
                "selected_C": best_C,
                "classifier": "L2 logistic regression",
                **metrics,
            }
        )
        prediction_table[name] = p_val

        selected_hyperparameters.append(
            {
                "model": name,
                "selected_C": best_C,
                "selection_metric": "vehicle-level tuning PR-AUC",
                "classifier": "L2 logistic regression",
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 3000,
                "seed": seed,
            }
        )

        save_control_model(
            model,
            checkpoints_dir
            / "controls"
            / f"{name}.joblib",
        )

    # Add the frozen proposed model for context only. It is not involved in
    # control-model hyperparameter selection.
    proposed_aligned = val_meta[
        ["vehicle_id"]
    ].merge(
        proposed_validation[
            [
                "vehicle_id",
                "raw_probability",
            ]
        ],
        on="vehicle_id",
        how="left",
        validate="one_to_one",
    )
    if proposed_aligned["raw_probability"].isna().any():
        raise RuntimeError(
            "Could not align frozen proposed-model validation predictions."
        )

    proposed_metrics = binary_ranking_metrics(
        y_val,
        proposed_aligned["raw_probability"],
    )
    metric_rows.append(
        {
            "model": "frozen_proposed_gru",
            "representation_dimension": int(
                cfg["model"]["fusion_hidden"]
            ),
            "selected_C": np.nan,
            "classifier": "frozen proposed GRU risk head",
            **proposed_metrics,
        }
    )
    prediction_table["frozen_proposed_gru"] = (
        proposed_aligned["raw_probability"].to_numpy()
    )

    metrics_df = pd.DataFrame(metric_rows)
    tuning_grid_df = pd.concat(
        tuning_tables,
        ignore_index=True,
    )
    selected_df = pd.DataFrame(
        selected_hyperparameters
    )

    dimensions_df = pd.DataFrame(
        [
            {
                "representation": "time_only",
                "dimension": 2,
                "contents": "prefix_time_step + prefix_length",
            },
            {
                "representation": "static_only",
                "dimension": int(static_dim),
                "contents": "training-fitted one-hot static specifications",
            },
            {
                "representation": "aggregate_dynamic",
                "dimension": int(X_train_agg.shape[1]),
                "contents": (
                    "mean/std/min/max/final/trend for each of "
                    f"{dynamic_dim} dynamic variables"
                ),
            },
            {
                "representation": "gru_temporal",
                "dimension": int(X_train_gru.shape[1]),
                "contents": "frozen h_i from the Stage 2E.1 GRU",
            },
            {
                "representation": "aggregate_dynamic_plus_static",
                "dimension": int(
                    X_train_agg.shape[1] + static_dim
                ),
                "contents": "aggregate dynamic summaries + static one-hot",
            },
            {
                "representation": "gru_temporal_plus_static",
                "dimension": int(
                    X_train_gru.shape[1] + static_dim
                ),
                "contents": "frozen GRU h_i + static one-hot",
            },
        ]
    )

    comparison_df = representation_comparison_summary(
        metrics_df
    )

    tuning_grid_df.to_csv(
        tables_dir / "temporal_control_tuning_grid.csv",
        index=False,
    )
    selected_df.to_csv(
        tables_dir
        / "temporal_control_selected_hyperparameters.csv",
        index=False,
    )
    dimensions_df.to_csv(
        tables_dir
        / "temporal_control_representation_dimensions.csv",
        index=False,
    )
    metrics_df.to_csv(
        tables_dir
        / "temporal_control_validation_metrics.csv",
        index=False,
    )
    comparison_df.to_csv(
        tables_dir
        / "temporal_control_pairwise_summary.csv",
        index=False,
    )
    prediction_table.to_csv(
        predictions_dir
        / "temporal_control_validation_predictions.csv",
        index=False,
    )

    # Feature matrices are saved as compressed NumPy files so the exact
    # reviewer control can be reproduced without re-streaming the raw CSV.
    np.savez_compressed(
        features_dir / "temporal_control_train_features.npz",
        vehicle_id=train_meta["vehicle_id"].to_numpy(dtype=np.int64),
        prefix_time_step=train_meta[
            "prefix_time_step"
        ].to_numpy(dtype=np.int64),
        label=y_train.astype(np.int8),
        aggregate_dynamic=X_train_agg,
        static=X_train_static,
        time_only=X_train_time,
        gru_temporal=X_train_gru,
    )
    np.savez_compressed(
        features_dir / "temporal_control_tuning_features.npz",
        vehicle_id=tune_meta["vehicle_id"].to_numpy(dtype=np.int64),
        prefix_time_step=tune_meta[
            "prefix_time_step"
        ].to_numpy(dtype=np.int64),
        label=y_tune.astype(np.int8),
        aggregate_dynamic=X_tune_agg,
        static=X_tune_static,
        time_only=X_tune_time,
        gru_temporal=X_tune_gru,
    )
    np.savez_compressed(
        features_dir / "temporal_control_validation_features.npz",
        vehicle_id=val_meta["vehicle_id"].to_numpy(dtype=np.int64),
        binary_label=y_val.astype(np.int8),
        aggregate_dynamic=X_val_agg,
        static=X_val_static,
        time_only=X_val_time,
        gru_temporal=X_val_gru,
    )

    summary = {
        "stage": "4A",
        "frozen_gru_best_epoch": int(
            frozen_checkpoint.get("best_epoch", -1)
        ),
        "model_training_samples": int(len(y_train)),
        "vehicle_level_tuning_samples": int(len(y_tune)),
        "validation_vehicles": int(len(y_val)),
        "validation_positives": int(y_val.sum()),
        "control_classifier": "L2 logistic regression",
        "C_grid": CONTROL_C_GRID,
        "validation_used_for_hyperparameter_selection": False,
        "metrics": metrics_df.to_dict(orient="records"),
        "pairwise": comparison_df.to_dict(orient="records"),
    }
    with (
        logs_dir / "stage4a_temporal_controls_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("-" * 78)
    print("STAGE 4A TEMPORAL CONTROL RESULTS")
    print("-" * 78)
    print(
        metrics_df[
            [
                "model",
                "representation_dimension",
                "selected_C",
                "auroc",
                "pr_auc",
                "enrichment",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print()
    print("Reviewer-isolation comparisons:")
    if len(comparison_df):
        print(
            comparison_df[
                [
                    "comparison",
                    "auroc_difference_a_minus_b",
                    "pr_auc_difference_a_minus_b",
                    "enrichment_difference_a_minus_b",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
    print()
    print(
        "All logistic-control hyperparameters were selected using the "
        "vehicle-level tuning landmarks only. Public validation was used "
        "only once for the displayed evaluation metrics."
    )
    print()
    print("Saved:")
    for path in [
        tables_dir / "temporal_control_tuning_grid.csv",
        tables_dir
        / "temporal_control_selected_hyperparameters.csv",
        tables_dir
        / "temporal_control_representation_dimensions.csv",
        tables_dir
        / "temporal_control_validation_metrics.csv",
        tables_dir
        / "temporal_control_pairwise_summary.csv",
        predictions_dir
        / "temporal_control_validation_predictions.csv",
        logs_dir / "stage4a_temporal_controls_summary.json",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
